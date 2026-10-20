// End-to-end ESN inference assertion test.
//
// Loads an ESN GGUF produced by scripts/esn_training.py and exercises the
// public llama C API, then asserts a set of *numerical* properties of the
// forward pass that go beyond "it compiled and ran":
//
//   A. GGUF loads: arch=esn, vocab > 0.
//   B. Context init allocates recurrent state of size reservoir_size
//      (indirectly, via a successful decode).
//   C. Multi-token prompt decode succeeds (unrolled recurrence path).
//   D. Single-token step decode succeeds (continues from prompt state).
//   E. Logits are all finite and have the expected dimensionality.
//   F. Determinism: feeding the same prompt into two freshly-initialised
//      contexts produces bit-identical logits.
//   G. Prompt sensitivity: two different prompts of the same length
//      produce different logits (the reservoir actually sees input).
//   H. State advancement: the logits at step t+1 differ from step t when
//      the input byte differs (the recurrence is actually running, i.e.
//      state is not frozen).
//   I. Softmax is well-defined: exp(logits - max) sums to a finite,
//      strictly-positive value.
//   J. Batch consistency: decoding a prompt in one batch with every
//      position as an output — alone, and alongside a second sequence in
//      the same batch — yields the same per-position logits as feeding it
//      one token at a time.
//
// Invocation (from the build dir):
//   bin/test-esn-inference <path-to-esn-model.gguf>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void silent_log(enum ggml_log_level, const char *, void *) {}

static llama_context * make_ctx(llama_model * model, uint32_t n_seq_max = 1) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 64;
    cparams.n_batch   = 64;
    cparams.n_ubatch  = 64;
    cparams.n_seq_max = n_seq_max;
    cparams.no_perf   = true;
    return llama_init_from_model(model, cparams);
}

// Decode a prompt. llama_batch_get_one (logits=nullptr) emits logits
// only at the last position in the batch, which is what we want: we read
// the single output row with llama_get_logits_ith(ctx, -1).
static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::vector<llama_token> t = tokens;
    llama_batch b = llama_batch_get_one(t.data(), (int32_t) t.size());
    return llama_decode(ctx, b) == 0;
}

static bool decode_step(llama_context * ctx, llama_token t) {
    llama_token tt = t;
    llama_batch b = llama_batch_get_one(&tt, 1);
    return llama_decode(ctx, b) == 0;
}

static std::vector<float> snapshot_logits(llama_context * ctx, int32_t n_vocab) {
    // -1 reads the last output row produced by the most recent decode.
    const float * p = llama_get_logits_ith(ctx, -1);
    assert(p != nullptr && "logits must be available after decode (index -1)");
    std::vector<float> out(n_vocab);
    std::memcpy(out.data(), p, (size_t) n_vocab * sizeof(float));
    return out;
}

static bool all_finite(const std::vector<float> & v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

static bool bit_identical(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    assert(a.size() == b.size());
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > m) m = d;
    }
    return m;
}

static double max_abs(const std::vector<float> & v) {
    double m = 0.0;
    for (float x : v) m = std::fmax(m, std::fabs((double) x));
    return m;
}

// Per-position logits from feeding tokens one decode call at a time.
static std::vector<std::vector<float>> stepwise_logits(llama_model * model, const std::vector<llama_token> & tokens,
                                                       int32_t n_vocab) {
    llama_context * ctx = make_ctx(model);
    assert(ctx && "J: init stepwise context");
    std::vector<std::vector<float>> out;
    for (llama_token t : tokens) {
        const bool ok = decode_step(ctx, t);
        assert(ok && "J: stepwise decode");
        out.push_back(snapshot_logits(ctx, n_vocab));
    }
    llama_free(ctx);
    return out;
}

// Decode all sequences in one batch with every position as an output and
// return the largest deviation from the stepwise reference, relative to the
// reference's magnitude.
static double batched_rel_error(llama_model * model, const std::vector<std::vector<llama_token>> & seqs,
                                const std::vector<std::vector<std::vector<float>>> & refs, int32_t n_vocab) {
    int32_t n_total = 0;
    for (const auto & s : seqs) n_total += (int32_t) s.size();

    llama_context * ctx = make_ctx(model, (uint32_t) seqs.size());
    assert(ctx && "J: init batched context");
    llama_batch batch = llama_batch_init(n_total, 0, 1);
    for (size_t s = 0; s < seqs.size(); ++s) {
        for (size_t t = 0; t < seqs[s].size(); ++t) {
            const int32_t i = batch.n_tokens++;
            batch.token[i]     = seqs[s][t];
            batch.pos[i]       = (llama_pos) t;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = (llama_seq_id) s;
            batch.logits[i]    = 1;
        }
    }
    const int32_t rc = llama_decode(ctx, batch);
    assert(rc == 0 && "J: batched decode");

    double worst = 0.0;
    int32_t i = 0;
    for (size_t s = 0; s < seqs.size(); ++s) {
        for (size_t t = 0; t < seqs[s].size(); ++t, ++i) {
            const float * p = llama_get_logits_ith(ctx, i);
            assert(p != nullptr && "J: batched logits");
            const std::vector<float> got(p, p + n_vocab);
            const auto & want = refs[s][t];
            worst = std::fmax(worst, max_abs_diff(got, want) / (1.0 + max_abs(want)));
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return worst;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <esn.gguf>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string model_path = argv[1];
    fprintf(stderr, "test-esn-inference: using '%s'\n", model_path.c_str());

    llama_log_set(silent_log, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap     = false;
    mparams.n_gpu_layers = 0;

    // --- A. load ---------------------------------------------------------
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "FAIL(A): load model\n"); return EXIT_FAILURE; }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    assert(n_vocab > 0 && "vocab must be non-empty");
    fprintf(stderr, "test-esn-inference: A.load OK (n_vocab=%d)\n", n_vocab);

    // Helper: build four prompts we'll reuse.
    const std::vector<llama_token> prompt_a      = { 1, 2, 3, 4 };
    const std::vector<llama_token> prompt_a_copy = { 1, 2, 3, 4 };
    const std::vector<llama_token> prompt_b      = { 5, 6, 7, 8 };
    const llama_token              step_x        = (llama_token) (10 % n_vocab);
    const llama_token              step_y        = (llama_token) (123 % n_vocab);

    // --- B. context + C. multi-token decode + E. finite logits -----------
    llama_context * ctx1 = make_ctx(model);
    if (!ctx1) { fprintf(stderr, "FAIL(B): init context\n"); llama_model_free(model); return EXIT_FAILURE; }
    if (!decode_tokens(ctx1, prompt_a)) { fprintf(stderr, "FAIL(C): decode prompt_a\n"); return EXIT_FAILURE; }
    auto logits_a_prompt = snapshot_logits(ctx1, n_vocab);
    assert(all_finite(logits_a_prompt) && "E: logits must be finite after prompt decode");
    fprintf(stderr, "test-esn-inference: B+C+E OK (%zu-token prompt → %d finite logits)\n",
            prompt_a.size(), n_vocab);

    // --- D. single-token step continues from prompt state ----------------
    if (!decode_step(ctx1, step_x)) {
        fprintf(stderr, "FAIL(D): single-token step decode\n");
        return EXIT_FAILURE;
    }
    auto logits_a_step_x = snapshot_logits(ctx1, n_vocab);
    assert(all_finite(logits_a_step_x) && "E: logits must be finite after step decode");

    // --- H. state advancement: logits after a *different* input byte must
    // not equal logits after the previous byte (unless the reservoir is
    // completely frozen, which is a bug).
    if (!decode_step(ctx1, step_y)) {
        fprintf(stderr, "FAIL(H): decode step_y\n");
        return EXIT_FAILURE;
    }
    auto logits_a_step_y = snapshot_logits(ctx1, n_vocab);
    const double adv_diff = max_abs_diff(logits_a_step_x, logits_a_step_y);
    assert(adv_diff > 0.0 && "H: state must advance — different input, same logits means recurrence is dead");
    fprintf(stderr, "test-esn-inference: D+H OK (step logits finite; max|Δ| step_x→step_y = %.6g)\n", adv_diff);

    llama_free(ctx1);

    // --- F. determinism: fresh context, same prompt → same logits --------
    llama_context * ctx2 = make_ctx(model);
    if (!ctx2) { fprintf(stderr, "FAIL(F): init second context\n"); llama_model_free(model); return EXIT_FAILURE; }
    if (!decode_tokens(ctx2, prompt_a_copy)) { fprintf(stderr, "FAIL(F): decode prompt_a_copy\n"); return EXIT_FAILURE; }
    auto logits_a_prompt_again = snapshot_logits(ctx2, n_vocab);
    assert(bit_identical(logits_a_prompt, logits_a_prompt_again) &&
           "F: deterministic forward pass — same prompt in a fresh context must produce identical logits");
    fprintf(stderr, "test-esn-inference: F OK (deterministic — bit-identical logits across two contexts)\n");
    llama_free(ctx2);

    // --- G. prompt sensitivity: different prompt → different logits ------
    llama_context * ctx3 = make_ctx(model);
    if (!ctx3) { fprintf(stderr, "FAIL(G): init third context\n"); llama_model_free(model); return EXIT_FAILURE; }
    if (!decode_tokens(ctx3, prompt_b)) { fprintf(stderr, "FAIL(G): decode prompt_b\n"); return EXIT_FAILURE; }
    auto logits_b_prompt = snapshot_logits(ctx3, n_vocab);
    const double sens_diff = max_abs_diff(logits_a_prompt, logits_b_prompt);
    assert(sens_diff > 0.0 && "G: prompt sensitivity — different prompt must produce different logits");
    fprintf(stderr, "test-esn-inference: G OK (prompt sensitivity: max|Δ| = %.6g)\n", sens_diff);
    llama_free(ctx3);

    // --- I. softmax is well-defined on the final logits ------------------
    float lmax = logits_a_prompt[0];
    for (int i = 1; i < n_vocab; ++i) if (logits_a_prompt[i] > lmax) lmax = logits_a_prompt[i];
    double z = 0.0;
    for (int i = 0; i < n_vocab; ++i) z += std::exp((double)(logits_a_prompt[i] - lmax));
    assert(std::isfinite(z) && z > 0.0 && "I: softmax denominator must be finite and positive");
    fprintf(stderr, "test-esn-inference: I OK (softmax Z = %.6g, lmax = %.6g)\n", z, (double) lmax);

    // --- J. batch consistency ---------------------------------------------
    // An odd, non-power-of-two length so the multi-token output stacking
    // sees uneven splits.
    std::vector<llama_token> seq_p, seq_q;
    for (int t = 0; t < 23; ++t) {
        seq_p.push_back((llama_token) ((7 * t + 3)  % n_vocab));
        seq_q.push_back((llama_token) ((11 * t + 5) % n_vocab));
    }
    const auto ref_p = stepwise_logits(model, seq_p, n_vocab);
    const auto ref_q = stepwise_logits(model, seq_q, n_vocab);
    const double err_one = batched_rel_error(model, { seq_p }, { ref_p }, n_vocab);
    const double err_two = batched_rel_error(model, { seq_p, seq_q }, { ref_p, ref_q }, n_vocab);
    // Batched and stepwise decodes may take different matmul kernels, so
    // allow rounding: ~1e-5 for an F32 reservoir, ~1e-3 for an F16 one
    // (its product rounds the state to F16). A misplaced row is off by O(1).
    assert(err_one < 1e-2 && "J: one-sequence batched logits must match stepwise decoding");
    assert(err_two < 1e-2 && "J: two-sequence batched logits must match stepwise decoding");
    fprintf(stderr, "test-esn-inference: J OK (batched vs stepwise, rel. max|Δ|: 1 seq %.3g, 2 seqs %.3g)\n",
            err_one, err_two);

    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "test-esn-inference: PASSED (A..J)\n");
    return EXIT_SUCCESS;
}
