// Online readout adaptation of an ESN context (the llama_esn_* API).
//
// Writes small ESN GGUFs with the gguf API (random reservoir, byte
// vocabulary; one with esn.online_learning.enabled, one without), decodes
// through the public llama API, and checks that feedback actually retrains
// the readout the next decode uses:
//
//   A. Non-ESN-adaptive and argument errors: -2 for a NULL context, -4 for an
//      ESN without online learning, -3 for feedback on a position that was
//      never decoded, for latency events and for out-of-range tokens.
//   B. Ridge (mode 0): one correction at a position moves the readout so the
//      logits there become T * one-hot(target) (T = esn.output_temperature),
//      to within lambda; the update returns the number of events consumed.
//   C. The adapted readout is per context: a second context on the same model
//      still produces the checkpoint logits.
//   D. llama_esn_reset_adaptation restores the checkpoint logits bit for bit.
//   E. RLS (mode 1) on the same single sample reaches the ridge solution.
//   F. SGD (mode 2): a correction raises the target's probability; a negative
//      reward lowers it. Least-squares modes refuse negative rewards (-7).
//   G. Graph reuse: single-token decodes after an update keep using the
//      readout as later updates change it in place.
//   H. The feedback buffer is bounded by esn.online_learning.buf_size (-5).
//
// With a model path argument the same flow runs on that model (it must have
// esn.online_learning.enabled) with checks that do not depend on knowing its
// temperature: after a ridge correction the target is the argmax.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

static constexpr uint32_t N_RES   = 64;
static constexpr uint32_t N_EMBD  = 16;
static constexpr uint32_t N_VOCAB = 256;
static constexpr float    TEMP    = 2.0f;

static void silent_log(enum ggml_log_level, const char *, void *) {}

static void write_model(const std::string & path, bool online) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str (g, "general.architecture", "esn");
    gguf_set_val_u32 (g, "esn.vocab_size", N_VOCAB);
    gguf_set_val_u32 (g, "esn.embedding_length", N_EMBD);
    gguf_set_val_u32 (g, "esn.context_length", 2048);
    gguf_set_val_u32 (g, "esn.block_count", 1);
    gguf_set_val_f32 (g, "esn.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_u32 (g, "esn.reservoir_size", N_RES);
    gguf_set_val_f32 (g, "esn.spectral_radius", 0.9f);
    gguf_set_val_f32 (g, "esn.sparsity", 1.0f);
    gguf_set_val_f32 (g, "esn.leaking_rate", 0.5f);
    gguf_set_val_f32 (g, "esn.input_scaling", 1.0f);
    gguf_set_val_f32 (g, "esn.output_temperature", TEMP);
    if (online) {
        gguf_set_val_bool(g, "esn.online_learning.enabled", true);
        gguf_set_val_u32 (g, "esn.online_learning.buf_size", 64);
        gguf_set_val_u32 (g, "esn.replay_window", 256);
    }

    std::vector<std::string> text(N_VOCAB);
    std::vector<const char *> tokens(N_VOCAB);
    std::vector<int32_t> types(N_VOCAB, 1);
    for (uint32_t b = 0; b < N_VOCAB; ++b) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\x%02x", b);
        text[b] = buf;
        tokens[b] = text[b].c_str();
    }
    gguf_set_val_str (g, "tokenizer.ggml.model", "rwkv");
    gguf_set_arr_str (g, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + 4 * (N_EMBD * N_VOCAB + N_EMBD * N_RES + N_RES * N_RES + N_RES + N_RES * N_VOCAB) + 4096,
                            nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto fill = [&](const char * name, int64_t ne0, int64_t ne1, float scale) {
        ggml_tensor * t = ne1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1) : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        ggml_set_name(t, name);
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            d[i] = scale < 0 ? 1.0f : scale * nd(rng);
        }
        gguf_add_tensor(g, t);
    };
    fill("token_embd.weight",            N_EMBD, N_VOCAB, 1.0f);
    fill("esn_input_weights.weight",     N_EMBD, N_RES,   0.5f / std::sqrt((float) N_EMBD));
    fill("esn_reservoir_weights.weight", N_RES,  N_RES,   0.9f / std::sqrt((float) N_RES));
    fill("output_norm.weight",           N_RES,  0,       -1.0f);
    fill("esn_output_weights.weight",    N_RES,  N_VOCAB, 0.1f);

    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    assert(ok);
    gguf_free(g);
    ggml_free(ctx);
}

// A rank-1 LoRA on the readout.
static void write_readout_lora(const std::string & path) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.type", "adapter");
    gguf_set_val_str(g, "general.architecture", "esn");
    gguf_set_val_str(g, "adapter.type", "lora");
    gguf_set_val_f32(g, "adapter.lora.alpha", 1.0f);

    ggml_init_params ip = { 4 * ggml_tensor_overhead() + 4 * (N_RES + N_VOCAB) + 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_RES, 1);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, N_VOCAB);
    ggml_set_name(a, "esn_output_weights.weight.lora_a");
    ggml_set_name(b, "esn_output_weights.weight.lora_b");
    for (int64_t i = 0; i < ggml_nelements(a); ++i) { ((float *) a->data)[i] = 0.01f; }
    for (int64_t i = 0; i < ggml_nelements(b); ++i) { ((float *) b->data)[i] = 0.01f * (float) (i % 7); }
    gguf_add_tensor(g, a);
    gguf_add_tensor(g, b);

    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    assert(ok);
    gguf_free(g);
    ggml_free(ctx);
}

static llama_context * make_ctx(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 128;
    cp.n_batch  = 128;
    cp.n_ubatch = 128;
    cp.no_perf  = true;
    return llama_init_from_model(model, cp);
}

// Decode tokens from position 0 with every position as an output and return
// the logits of all positions.
static std::vector<std::vector<float>> decode_all(llama_context * ctx, const std::vector<llama_token> & toks) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i];
        b.pos[i] = (llama_pos) i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, b);
    assert(rc == 0);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    std::vector<std::vector<float>> out(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        const float * l = llama_get_logits_ith(ctx, (int32_t) i);
        out[i].assign(l, l + n_vocab);
    }
    llama_batch_free(b);
    return out;
}

static double prob(const std::vector<float> & z, int tok) {
    const float m = *std::max_element(z.begin(), z.end());
    double s = 0.0;
    for (float v : z) {
        s += std::exp((double) (v - m));
    }
    return std::exp((double) (z[tok] - m)) / s;
}

static llama_esn_feedback_event event(llama_esn_feedback_type type, llama_pos pos, const llama_token * tok, float signal = 0.0f) {
    return { type, 0, pos, signal, tok, 1 };
}

static llama_esn_online_params mode_params(uint32_t mode, float lr = 0.0f, float reg = 1e-4f) {
    llama_esn_online_params p = llama_esn_online_params_default();
    p.enabled        = true;
    p.update_mode    = mode;
    p.learning_rate  = lr;
    p.regularization = reg;
    return p;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::fabs(a[i] - b[i]));
    }
    return d;
}

int main(int argc, char ** argv) {
    if (getenv("ESN_TEST_LOG") == nullptr) {
        llama_log_set(silent_log, nullptr);
    }
    llama_backend_init();
    // a GGML_BACKEND_DL build registers no backend until it is loaded
    ggml_backend_load_all();

    const bool synthetic = argc < 2;
    const auto dir = std::filesystem::temp_directory_path();
    const std::string path_on  = synthetic ? (dir / "test-esn-online.gguf").string() : argv[1];
    const std::string path_off = (dir / "test-esn-online-off.gguf").string();
    if (synthetic) {
        write_model(path_on, true);
    }
    write_model(path_off, false);

    llama_model_params mp = llama_model_default_params();
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(path_on.c_str(), mp);
    llama_model * plain = llama_model_load_from_file(path_off.c_str(), mp);
    assert(model && plain);
    assert(llama_esn_is_adaptive(model) && !llama_esn_is_adaptive(plain));
    assert(llama_esn_reservoir_size(plain) == N_RES && llama_esn_n_levels(plain) == 1);
    assert(llama_esn_model_online_params(model).enabled && !llama_esn_model_online_params(plain).enabled);
    if (synthetic) {
        const llama_esn_online_params mp_on = llama_esn_model_online_params(model);
        assert(mp_on.update_mode == 0 && mp_on.replay_window == 256 && mp_on.freeze_reservoir);
    }

    printf("test-esn-online: %s\n", synthetic ? "synthetic model" : path_on.c_str());

    const std::vector<llama_token> prompt = { 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd' };
    const llama_pos   at     = 4;
    const llama_token target = 'z';

    // A
    {
        llama_context * ctx_plain = make_ctx(plain);
        llama_esn_feedback_event ev = event(LLAMA_ESN_FEEDBACK_CORRECTION, 0, &target);
        llama_esn_online_params p = mode_params(0);
        assert(llama_esn_submit_feedback(nullptr, &ev, 1) == -2);
        assert(llama_esn_update_readout(nullptr, &p) == -2);
        assert(llama_esn_submit_feedback(ctx_plain, &ev, 1) == -4);
        assert(llama_esn_update_readout(ctx_plain, &p) == -4);
        llama_esn_reset_adaptation(ctx_plain);
        llama_free(ctx_plain);
        printf("  A1 errors on NULL / non-adaptive contexts: ok\n");
    }

    llama_context * ctx  = make_ctx(model);
    llama_context * ctx2 = make_ctx(model);
    {
        llama_esn_feedback_event ev = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &target);
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == -3);   // nothing decoded yet
    }

    const auto base = decode_all(ctx, prompt);
    {
        const llama_token bad = (llama_token) N_VOCAB;
        llama_esn_feedback_event lat = event(LLAMA_ESN_FEEDBACK_LATENCY, at, &target, 12.0f);
        llama_esn_feedback_event oob = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &bad);
        llama_esn_feedback_event far = event(LLAMA_ESN_FEEDBACK_CORRECTION, (llama_pos) prompt.size(), &target);
        llama_esn_feedback_event ok  = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &target);
        llama_esn_feedback_event batch[2] = { ok, oob };
        assert(llama_esn_submit_feedback(ctx, &lat, 1) == -3);
        assert(llama_esn_submit_feedback(ctx, &oob, 1) == -3);
        assert(llama_esn_submit_feedback(ctx, &far, 1) == -3);
        assert(llama_esn_submit_feedback(ctx, batch, 2) == -3); // all or nothing
        llama_esn_online_params p = mode_params(0);
        assert(llama_esn_update_readout(ctx, &p) == 0);          // nothing pending
        p.freeze_reservoir = false;
        assert(llama_esn_update_readout(ctx, &p) == -3);
        printf("  A2 invalid feedback and parameters are rejected: ok\n");
    }

    // B: ridge
    std::vector<std::vector<float>> ridge;
    {
        llama_esn_feedback_event ev = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &target);
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == 0);
        llama_esn_online_params off = mode_params(0);
        off.enabled = false;
        assert(llama_esn_update_readout(ctx, &off) == 0);
        assert(max_abs_diff(decode_all(ctx, prompt)[at], base[at]) == 0.0f);

        llama_esn_online_params p = mode_params(0, 0.0f, 1e-4f);
        assert(llama_esn_update_readout(ctx, &p) == 1);
        ridge = decode_all(ctx, prompt);
        const auto & z = ridge[at];
        int argmax = (int) (std::max_element(z.begin(), z.end()) - z.begin());
        assert(argmax == target);
        if (synthetic) {
            // |x|^2 = N_RES after the RMS norm, so the residual shrinks by
            // lambda / (N_RES + lambda)
            for (uint32_t v = 0; v < N_VOCAB; ++v) {
                const float want = v == (uint32_t) target ? TEMP : 0.0f;
                assert(std::fabs(z[v] - want) < 1e-3f);
            }
        }
        assert(max_abs_diff(ridge[at + 1], base[at + 1]) > 0.0f); // the readout changed everywhere
        printf("  B  ridge correction: logit[target] %.4f -> %.4f, p %.4f -> %.4f: ok\n",
                base[at][target], z[target], prob(base[at], target), prob(z, target));
    }

    // C
    {
        const auto other = decode_all(ctx2, prompt);
        for (size_t i = 0; i < prompt.size(); ++i) {
            assert(max_abs_diff(other[i], base[i]) == 0.0f);
        }
        printf("  C  a second context keeps the checkpoint readout: ok\n");
    }

    // D
    {
        llama_esn_reset_adaptation(ctx);
        const auto again = decode_all(ctx, prompt);
        for (size_t i = 0; i < prompt.size(); ++i) {
            assert(max_abs_diff(again[i], base[i]) == 0.0f);
        }
        printf("  D  reset restores the checkpoint logits exactly: ok\n");
    }

    // E: RLS on the same sample
    {
        llama_esn_feedback_event ev = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &target);
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == 0);
        llama_esn_online_params p = mode_params(1, 0.0f, 1e-4f);
        assert(llama_esn_update_readout(ctx, &p) == 1);
        const auto rls = decode_all(ctx, prompt);
        for (size_t i = 0; i < prompt.size(); ++i) {
            assert(max_abs_diff(rls[i], ridge[i]) < 1e-3f);
        }
        printf("  E  RLS reaches the ridge solution for one sample: ok\n");
        llama_esn_reset_adaptation(ctx);
    }

    // F: SGD
    {
        decode_all(ctx, prompt);
        llama_esn_feedback_event ev = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &target);
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == 0);
        llama_esn_online_params p = mode_params(2, 0.02f, 0.0f);
        assert(llama_esn_update_readout(ctx, &p) == 1);
        const double p_up = prob(decode_all(ctx, prompt)[at], target);
        assert(p_up > prob(base[at], target));

        const llama_token emitted = 'q';
        llama_esn_feedback_event pen = event(LLAMA_ESN_FEEDBACK_REWARD, at, &emitted, -1.0f);
        const double q_before = prob(decode_all(ctx, prompt)[at], emitted);
        assert(llama_esn_submit_feedback(ctx, &pen, 1) == 0);
        llama_esn_online_params ls = mode_params(0);
        assert(llama_esn_update_readout(ctx, &ls) == -7);        // stays pending
        assert(llama_esn_update_readout(ctx, &p) == 1);
        const double q_after = prob(decode_all(ctx, prompt)[at], emitted);
        assert(q_after < q_before);
        printf("  F  SGD: p[target] %.4f -> %.4f; negative reward p %.4f -> %.4f: ok\n",
                prob(base[at], target), p_up, q_before, q_after);
        llama_esn_reset_adaptation(ctx);
    }

    // G: in-place readout updates under graph reuse
    {
        decode_all(ctx, prompt);
        llama_esn_feedback_event ev = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &target);
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == 0);
        llama_esn_online_params p = mode_params(2, 0.02f, 0.0f);
        assert(llama_esn_update_readout(ctx, &p) == 1);

        // one token at a time from pos 0: same-shaped graphs, reused
        auto step_all = [&]() {
            llama_memory_clear(llama_get_memory(ctx), true);
            std::vector<float> last;
            for (size_t i = 0; i <= (size_t) at; ++i) {
                llama_token t = prompt[i];
                assert(llama_decode(ctx, llama_batch_get_one(&t, 1)) == 0);
            }
            const float * l = llama_get_logits_ith(ctx, -1);
            last.assign(l, l + N_VOCAB);
            return last;
        };
        const auto first = step_all();
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == 0);
        assert(llama_esn_update_readout(ctx, &p) == 1);
        const int32_t reused = llama_perf_context(ctx).n_reused;
        const auto second = step_all();
        if (getenv("LLAMA_GRAPH_REUSE_DISABLE") == nullptr) {
            assert(llama_perf_context(ctx).n_reused > reused);
        }
        assert(prob(second, target) > prob(first, target));
        llama_esn_reset_adaptation(ctx);
        assert(max_abs_diff(step_all(), base[at]) < 1e-5f);
        printf("  G  updates apply to reused graphs: ok\n");
    }

    // H
    {
        std::vector<llama_token> longp;
        for (int i = 0; i < 70; ++i) {
            longp.push_back((llama_token) ('a' + i % 26));
        }
        decode_all(ctx, longp);
        std::vector<llama_token> tg(65, target);
        llama_esn_feedback_event ev = { LLAMA_ESN_FEEDBACK_CORRECTION, 0, 0, 0.0f, tg.data(), 65 };
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == -5);
        ev.n_target_tokens = 64;
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == 0);
        llama_esn_online_params p = mode_params(0, 0.0f, 1e-2f);
        assert(llama_esn_update_readout(ctx, &p) == 1);
        printf("  H  the feedback buffer is bounded: ok\n");
    }

    // I: recordings follow the sequence as it is rewound or cleared
    llama_esn_reset_adaptation(ctx);
    {
        llama_esn_online_params p = mode_params(0);
        decode_all(ctx, prompt);

        // a clear with nothing decoded after it: every position is gone
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_esn_feedback_event ev = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &target);
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == -3);

        // re-decoded with an output only at the last position: the earlier
        // positions were decoded again, so their old recordings are stale
        llama_batch b = llama_batch_init((int32_t) prompt.size(), 0, 1);
        for (size_t i = 0; i < prompt.size(); ++i) {
            b.token[i] = prompt[prompt.size() - 1 - i];
            b.pos[i] = (llama_pos) i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = i + 1 == prompt.size();
        }
        b.n_tokens = (int32_t) prompt.size();
        assert(llama_decode(ctx, b) == 0);
        llama_batch_free(b);
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == -3);
        llama_esn_feedback_event last = event(LLAMA_ESN_FEEDBACK_CORRECTION, (llama_pos) prompt.size() - 1, &target);
        assert(llama_esn_submit_feedback(ctx, &last, 1) == 0);
        assert(llama_esn_update_readout(ctx, &p) == 1);
        llama_esn_reset_adaptation(ctx);
        printf("  I  rewound and cleared positions cannot be trained on: ok\n");
    }

    // J: a singular solve fails without changing anything
    {
        decode_all(ctx, prompt);
        llama_esn_feedback_event ev = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &target);
        llama_esn_feedback_event twice[2] = { ev, ev };
        assert(llama_esn_submit_feedback(ctx, twice, 2) == 0);
        for (uint32_t mode : { 0u, 1u }) {
            // two identical samples: lambda below the rounding of |x|^2 leaves
            // the second pivot at zero
            llama_esn_online_params tiny = mode_params(mode, 0.0f, 1e-30f);
            const int32_t rc = llama_esn_update_readout(ctx, &tiny);
            assert(rc == -9 || (mode == 1 && rc == 2));
            if (rc == 2) {
                // RLS stays stable here; start over for the ridge check
                llama_esn_reset_adaptation(ctx);
                assert(llama_esn_submit_feedback(ctx, twice, 2) == 0);
                continue;
            }
            assert(max_abs_diff(decode_all(ctx, prompt)[at], base[at]) == 0.0f);
        }
        llama_esn_online_params p = mode_params(0, 0.0f, 1e-4f);
        assert(llama_esn_update_readout(ctx, &p) == 2);           // the samples were kept
        llama_esn_reset_adaptation(ctx);
        printf("  J  a singular least-squares solve is refused and changes nothing: ok\n");
    }

    // K: a LoRA on the readout and an adapted readout exclude each other
    if (synthetic) {
        const std::string path_lora = (dir / "test-esn-online-lora.gguf").string();
        write_readout_lora(path_lora);
        llama_adapter_lora * lora = llama_adapter_lora_init(model, path_lora.c_str());
        assert(lora != nullptr);

        decode_all(ctx, prompt);
        llama_esn_feedback_event ev = event(LLAMA_ESN_FEEDBACK_CORRECTION, at, &target);
        llama_esn_online_params p = mode_params(0);
        assert(llama_set_adapter_lora(ctx, lora, 1.0f) == 0);
        const auto with_lora = decode_all(ctx, prompt);
        assert(max_abs_diff(with_lora[at], base[at]) > 0.0f);     // the LoRA applies
        assert(llama_esn_submit_feedback(ctx, &ev, 1) == 0);
        assert(llama_esn_update_readout(ctx, &p) == -10);
        assert(llama_rm_adapter_lora(ctx, lora) == 0);
        assert(llama_esn_update_readout(ctx, &p) == 1);
        assert(llama_set_adapter_lora(ctx, lora, 1.0f) == -1);   // refused while adapted
        llama_esn_reset_adaptation(ctx);
        assert(llama_set_adapter_lora(ctx, lora, 1.0f) == 0);
        assert(llama_rm_adapter_lora(ctx, lora) == 0);

        llama_adapter_lora_free(lora);
        std::filesystem::remove(path_lora);
        printf("  K  a readout LoRA and online adaptation are mutually exclusive: ok\n");
    }

    llama_free(ctx);
    llama_free(ctx2);
    llama_model_free(model);
    llama_model_free(plain);
    if (synthetic) {
        std::filesystem::remove(path_on);
    }
    std::filesystem::remove(path_off);
    llama_backend_free();

    printf("test-esn-online: PASSED (A..K)\n");
    return 0;
}
