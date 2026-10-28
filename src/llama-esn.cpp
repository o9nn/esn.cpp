#include "llama-esn.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

//
// helpers
//

// Run fn(i0, i1) over [0, n) split across up to n_threads threads.
template <typename F>
static void esn_parallel_for(int n_threads, int64_t n, F && fn) {
    n_threads = (int) std::max<int64_t>(1, std::min<int64_t>(n_threads, n / 64));
    if (n_threads <= 1) {
        fn((int64_t) 0, n);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(n_threads - 1);
    const int64_t chunk = (n + n_threads - 1) / n_threads;
    for (int t = 1; t < n_threads; ++t) {
        const int64_t i0 = t * chunk;
        const int64_t i1 = std::min(n, i0 + chunk);
        if (i0 >= i1) {
            break;
        }
        workers.emplace_back([&fn, i0, i1]() { fn(i0, i1); });
    }
    fn((int64_t) 0, std::min(n, chunk));
    for (auto & w : workers) {
        w.join();
    }
}

// Eight independent partial sums, so the loop vectorizes without -ffast-math.
static float esn_dot(const float * a, const float * b, int64_t n) {
    float acc[8] = { 0 };
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        for (int k = 0; k < 8; ++k) {
            acc[k] += a[i + k] * b[i + k];
        }
    }
    float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

static void esn_axpy(float * y, float a, const float * x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

//
// llama_esn_adapter
//

llama_esn_adapter::llama_esn_adapter(const llama_model & model) :
    model(model),
    n_res  (model.hparams.esn_reservoir_size),
    n_vocab(model.output->ne[1]),
    recorded_cap(std::max<uint32_t>(1, model.hparams.esn_replay_window)),
    pending_cap (std::max<uint32_t>(1, model.hparams.esn_online_buffer_size)) {
    GGML_ASSERT(model.output && model.output->ne[0] == n_res);
}

llama_esn_adapter::~llama_esn_adapter() = default;

void llama_esn_adapter::capture(const llama_ubatch & ubatch, uint32_t n_outputs, const float * rows) {
    uint32_t j = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const bool is_output = n_outputs == ubatch.n_tokens || (n_outputs > 0 && ubatch.output[i]);
        const float * x = is_output ? rows + (size_t) j++ * n_res : nullptr;
        const llama_pos pos = ubatch.pos[i];
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            auto & q = recorded[ubatch.seq_id[i][s]];
            // decoding a position at or before the newest recorded one means
            // the sequence was rewound (seq_rm, memory clear): what was
            // recorded from there on belongs to tokens that are gone, whether
            // or not this token produces an output
            while (!q.empty() && q.back().pos >= pos) {
                q.pop_back();
            }
            if (x == nullptr) {
                continue;
            }
            q.push_back({ pos, std::vector<float>(x, x + n_res) });
            while (q.size() > recorded_cap) {
                q.pop_front();
            }
        }
    }
    GGML_ASSERT(j == n_outputs);
}

const llama_esn_adapter::captured * llama_esn_adapter::find(llama_seq_id seq, llama_pos pos) const {
    const auto it = recorded.find(seq);
    if (it == recorded.end()) {
        return nullptr;
    }
    const auto & q = it->second;
    const auto c = std::lower_bound(q.begin(), q.end(), pos,
            [](const captured & a, llama_pos p) { return a.pos < p; });
    return c != q.end() && c->pos == pos ? &*c : nullptr;
}

int32_t llama_esn_adapter::submit(const llama_esn_feedback_event * events, int32_t n_events,
                                  const std::function<llama_pos(llama_seq_id)> & pos_max) {
    if (n_events < 0 || (n_events > 0 && events == nullptr)) {
        return -3;
    }

    // validate everything before taking anything: a batch is all or nothing
    std::vector<sample> fresh;
    for (int32_t e = 0; e < n_events; ++e) {
        const auto & ev = events[e];

        float weight = 1.0f;
        switch (ev.type) {
            case LLAMA_ESN_FEEDBACK_CORRECTION:
            case LLAMA_ESN_FEEDBACK_TOOL_RESULT:
                weight = 1.0f;
                break;
            case LLAMA_ESN_FEEDBACK_REWARD:
                if (!std::isfinite(ev.signal)) {
                    LLAMA_LOG_ERROR("%s: event %d: reward signal is not finite\n", __func__, e);
                    return -3;
                }
                weight = ev.signal;
                break;
            case LLAMA_ESN_FEEDBACK_CONFIDENCE:
                if (!(ev.signal >= 0.0f && ev.signal <= 1.0f)) {
                    LLAMA_LOG_ERROR("%s: event %d: confidence must be in [0, 1], got %f\n", __func__, e, ev.signal);
                    return -3;
                }
                weight = ev.signal;
                break;
            case LLAMA_ESN_FEEDBACK_LATENCY:
                LLAMA_LOG_ERROR("%s: event %d: latency feedback carries no token target and the readout "
                        "does not determine latency; it cannot train the readout\n", __func__, e);
                return -3;
            default:
                LLAMA_LOG_ERROR("%s: event %d: unknown feedback type %d\n", __func__, e, (int) ev.type);
                return -3;
        }

        if (ev.n_target_tokens <= 0 || ev.target_tokens == nullptr) {
            LLAMA_LOG_ERROR("%s: event %d: target_tokens is required\n", __func__, e);
            return -3;
        }

        for (int32_t k = 0; k < ev.n_target_tokens; ++k) {
            const llama_token tok = ev.target_tokens[k];
            if (tok < 0 || tok >= n_vocab) {
                LLAMA_LOG_ERROR("%s: event %d: target token %d is out of range\n", __func__, e, tok);
                return -3;
            }
            const captured * c = ev.pos + k <= pos_max(ev.seq_id) ? find(ev.seq_id, ev.pos + k) : nullptr;
            if (c == nullptr) {
                LLAMA_LOG_ERROR("%s: event %d: no readout input recorded for seq %d pos %d (the position "
                        "was not decoded with an output, was rewound or cleared, or is older than the last %zu)\n",
                        __func__, e, ev.seq_id, ev.pos + k, recorded_cap);
                return -3;
            }
            // checked before each copy, so an oversized batch fails before
            // it allocates more than the buffer could hold
            if (pending.size() + fresh.size() >= pending_cap) {
                LLAMA_LOG_ERROR("%s: event %d: the batch exceeds the feedback buffer (%zu pending of %zu); "
                        "call llama_esn_update_readout first\n", __func__, e, pending.size(), pending_cap);
                return -5;
            }
            fresh.push_back({ c->x, tok, weight });
        }
    }

    for (auto & s : fresh) {
        pending.push_back(std::move(s));
    }
    pending_events += n_events;

    return 0;
}

float * llama_esn_adapter::w_data() const {
    return (float *) w->data;
}

void llama_esn_adapter::ensure_readout() {
    if (w != nullptr) {
        return;
    }

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        throw std::runtime_error("ESN: online adaptation needs the CPU backend");
    }

    // the checkpoint, dequantized
    const ggml_tensor * out = model.output;
    if (out->buffer == nullptr) {
        throw std::runtime_error("ESN: the readout has no data");
    }
    // the CPU's extra buffer types (REPACK, AMX, ...) hold weights in kernel
    // layouts and some cannot be read back at all
    {
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(out->buffer);
        auto * get_extra = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu), "ggml_backend_dev_get_extra_bufts");
        for (ggml_backend_buffer_type_t * e = get_extra ? get_extra(cpu) : nullptr; e && *e; ++e) {
            if (*e == buft) {
                throw std::runtime_error(std::string("ESN: the readout is stored in the ") + ggml_backend_buft_name(buft) +
                        " buffer for the CPU kernels and cannot be read back; load the model with "
                        "use_extra_bufts = false (--no-repack) to adapt it");
            }
        }
    }
    std::vector<uint8_t> raw(ggml_nbytes(out));
    ggml_backend_tensor_get(out, raw.data(), 0, raw.size());

    w0.resize((size_t) n_vocab * n_res);
    const size_t row_size = ggml_row_size(out->type, n_res);
    if (out->type == GGML_TYPE_F32) {
        memcpy(w0.data(), raw.data(), w0.size() * sizeof(float));
    } else {
        const auto * traits = ggml_get_type_traits(out->type);
        if (traits->to_float == nullptr) {
            throw std::runtime_error(std::string("ESN: cannot read a readout of type ") + ggml_type_name(out->type));
        }
        for (int64_t v = 0; v < n_vocab; ++v) {
            traits->to_float(raw.data() + v * row_size, w0.data() + v * n_res, n_res);
        }
    }

    // built in locals and committed only once fully allocated, so a failure
    // leaves the adapter as it was and a later update can retry
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("ESN: failed to create the adapted readout");
    }
    ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_res, n_vocab);
    ggml_set_name(t, "esn_adapted_readout");

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(cpu)));
    if (!buf) {
        throw std::runtime_error("ESN: failed to allocate the adapted readout");
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(buf.get()));
    memcpy(t->data, w0.data(), w0.size() * sizeof(float));

    ctx_w = std::move(ctx);
    buf_w = std::move(buf);
    w     = t;
}

int32_t llama_esn_adapter::update(const llama_esn_online_params & p, int n_threads) {
    if (!p.enabled) {
        return 0;
    }

    const bool finite = std::isfinite(p.learning_rate) && std::isfinite(p.regularization) && std::isfinite(p.decay_rate);
    if (!finite || p.update_mode > 2 || !p.freeze_reservoir || p.replay_window == 0 ||
        p.decay_rate < 0.0f || p.decay_rate >= 1.0f || p.regularization < 0.0f ||
        (p.update_mode != 2 && p.regularization <= 0.0f) ||
        (p.update_mode == 2 && p.learning_rate <= 0.0f)) {
        LLAMA_LOG_ERROR("%s: invalid parameters (mode %u, lr %g, reg %g, decay %g, window %u, freeze_reservoir %d)\n",
                __func__, p.update_mode, p.learning_rate, p.regularization, p.decay_rate, p.replay_window,
                (int) p.freeze_reservoir);
        return -3;
    }

    if (pending.empty()) {
        return 0;
    }

    const float temperature = model.hparams.esn_output_temperature;
    if (p.update_mode != 2) {
        if (temperature <= 0.0f) {
            LLAMA_LOG_ERROR("%s: least-squares updates need the scale the readout was fit to, and this "
                    "model has no esn.output_temperature; re-export it with scripts/esn_training.py "
                    "or use SGD (mode 2)\n", __func__);
            return -6;
        }
        const auto negative = [](const sample & s) { return s.weight < 0.0f; };
        bool has_negative = std::any_of(pending.begin(), pending.end(), negative);
        if (p.update_mode == 0) {
            const size_t n_hist = std::min<size_t>(history.size(), p.replay_window > pending.size() ? p.replay_window - pending.size() : 0);
            has_negative = has_negative || std::any_of(history.end() - n_hist, history.end(), negative);
        }
        if (has_negative) {
            LLAMA_LOG_ERROR("%s: negative-reward samples have no least-squares target; use SGD (mode 2)\n", __func__);
            return -7;
        }
    }

    ensure_readout();

    bool ok = true;
    switch (p.update_mode) {
        case 0: ok = update_ridge(p.regularization, p.decay_rate, p.replay_window, temperature, n_threads); break;
        case 1: ok = update_rls  (p.regularization, p.decay_rate, temperature);                           break;
        case 2:      update_sgd  (p.learning_rate, p.regularization);                                     break;
    }
    if (!ok) {
        LLAMA_LOG_ERROR("%s: the %s solve is numerically singular at regularization %g; the readout and the "
                "pending feedback are unchanged (raise the regularization)\n", __func__,
                p.update_mode == 0 ? "ridge" : "RLS", p.regularization);
        return -9;
    }
    last_mode = (int) p.update_mode;
    active    = true;

    for (auto & s : pending) {
        history.push_back(std::move(s));
    }
    while (history.size() > p.replay_window) {
        history.pop_front();
    }
    pending.clear();

    const int32_t consumed = pending_events;
    pending_events = 0;
    return consumed;
}

bool llama_esn_adapter::update_ridge(float lambda, float decay, uint32_t window, float temperature, int n_threads) {
    // the newest `window` samples of history + pending
    std::vector<const sample *> set;
    for (const auto & s : history) {
        set.push_back(&s);
    }
    for (const auto & s : pending) {
        set.push_back(&s);
    }
    if (set.size() > window) {
        set.erase(set.begin(), set.end() - window);
    }
    const int64_t n = (int64_t) set.size();

    // S[i] = sqrt(w_i) x_i, with w_i decayed by age (0 for the newest)
    std::vector<float> S((size_t) n * n_res);
    std::vector<float> sw(n);
    for (int64_t i = 0; i < n; ++i) {
        const double age = (double) (n - 1 - i);
        sw[i] = (float) std::sqrt((double) set[i]->weight * std::pow(1.0 - (double) decay, age));
        for (int64_t r = 0; r < n_res; ++r) {
            S[i * n_res + r] = sw[i] * set[i]->x[r];
        }
    }

    // G = S S^T + lambda I, Cholesky-factored in place (lower triangle)
    std::vector<double> G((size_t) n * n);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            double g = 0.0;
            const float * a = &S[i * n_res];
            const float * b = &S[j * n_res];
            for (int64_t r = 0; r < n_res; ++r) {
                g += (double) a[r] * b[r];
            }
            G[i * n + j] = g + (i == j ? (double) lambda : 0.0);
        }
    }
    for (int64_t j = 0; j < n; ++j) {
        const double diag = G[j * n + j];
        double d = diag;
        for (int64_t k = 0; k < j; ++k) {
            d -= G[j * n + k] * G[j * n + k];
        }
        // lambda > 0 makes G positive definite in exact arithmetic, but a
        // lambda below the rounding of S S^T leaves a pivot that is only
        // rounding noise: treat pivots under 1e-12 of their diagonal as zero
        if (!(d > 1e-12 * diag) || !std::isfinite(d)) {
            return false;
        }
        d = std::sqrt(d);
        G[j * n + j] = d;
        for (int64_t i = j + 1; i < n; ++i) {
            double s = G[i * n + j];
            for (int64_t k = 0; k < j; ++k) {
                s -= G[i * n + k] * G[j * n + k];
            }
            G[i * n + j] = s / d;
        }
    }

    // per output row v: e = sqrt(w) (T y_v - W0_v x), a = G^-1 e,
    // W_v = W0_v + sum_i a_i S_i
    float * W = w_data();
    esn_parallel_for(n_threads, n_vocab, [&](int64_t v0, int64_t v1) {
        std::vector<double> a(n);
        for (int64_t v = v0; v < v1; ++v) {
            const float * w0v = &w0[v * n_res];
            for (int64_t i = 0; i < n; ++i) {
                const double target = set[i]->token == v ? (double) sw[i] * temperature : 0.0;
                a[i] = target - esn_dot(w0v, &S[i * n_res], n_res);
            }
            for (int64_t i = 0; i < n; ++i) {          // L y = e
                double s = a[i];
                for (int64_t k = 0; k < i; ++k) {
                    s -= G[i * n + k] * a[k];
                }
                a[i] = s / G[i * n + i];
            }
            for (int64_t i = n - 1; i >= 0; --i) {     // L^T a = y
                double s = a[i];
                for (int64_t k = i + 1; k < n; ++k) {
                    s -= G[k * n + i] * a[k];
                }
                a[i] = s / G[i * n + i];
            }
            float * wv = W + v * n_res;
            memcpy(wv, w0v, n_res * sizeof(float));
            for (int64_t i = 0; i < n; ++i) {
                esn_axpy(wv, (float) a[i], &S[i * n_res], n_res);
            }
        }
    });
    return true;
}

bool llama_esn_adapter::update_rls(float lambda, float decay, float temperature) {
    const int64_t R = n_res;

    // restored if a step loses positive definiteness, so a failed update
    // changes nothing
    std::vector<float>  w_saved(w_data(), w_data() + (size_t) n_vocab * R);
    std::vector<double> p_saved = rls_p;
    const float lambda_saved = rls_lambda;
    const auto rollback = [&]() {
        memcpy(w_data(), w_saved.data(), w_saved.size() * sizeof(float));
        rls_p      = std::move(p_saved);
        rls_lambda = lambda_saved;
        return false;
    };

    // P is the inverse of lambda I + sum w x x^T for samples since the last
    // restart; it restarts at I / lambda (a prior centered on the current
    // readout) when lambda changes or another mode moved the readout
    if (rls_p.empty() || rls_lambda != lambda || last_mode != 1) {
        rls_p.assign((size_t) R * R, 0.0);
        for (int64_t r = 0; r < R; ++r) {
            rls_p[r * R + r] = 1.0 / lambda;
        }
        rls_lambda = lambda;
    }

    const double f = 1.0 - (double) decay;
    float * W = w_data();
    std::vector<double> g(R);
    std::vector<float>  k(R);

    for (const auto & s : pending) {
        if (s.weight <= 0.0f) {
            continue;
        }
        const float * x = s.x.data();

        double xg = 0.0;
        for (int64_t i = 0; i < R; ++i) {
            const double * pi = &rls_p[i * R];
            double acc = 0.0;
            for (int64_t j = 0; j < R; ++j) {
                acc += pi[j] * x[j];
            }
            g[i] = acc;
            xg  += acc * x[i];
        }
        const double denom = f / s.weight + xg;
        // P is positive definite, so xg >= 0; anything else is rounding loss
        if (!(xg >= 0.0) || !std::isfinite(denom)) {
            return rollback();
        }
        for (int64_t i = 0; i < R; ++i) {
            k[i] = (float) (g[i] / denom);
        }

        for (int64_t v = 0; v < n_vocab; ++v) {
            float * wv = W + v * R;
            const float e = (s.token == v ? temperature : 0.0f) - esn_dot(wv, x, R);
            esn_axpy(wv, e, k.data(), R);
        }

        for (int64_t i = 0; i < R; ++i) {
            double * pi = &rls_p[i * R];
            const double ki = g[i] / denom;
            for (int64_t j = 0; j < R; ++j) {
                pi[j] = (pi[j] - ki * g[j]) / f;
            }
        }
    }
    return true;
}

void llama_esn_adapter::update_sgd(float lr, float reg) {
    const int64_t R = n_res;
    float * W = w_data();
    std::vector<float> z(n_vocab);

    for (const auto & s : pending) {
        const float * x = s.x.data();

        float zmax = -INFINITY;
        for (int64_t v = 0; v < n_vocab; ++v) {
            z[v] = esn_dot(W + v * R, x, R);
            zmax = std::max(zmax, z[v]);
        }
        double sum = 0.0;
        for (int64_t v = 0; v < n_vocab; ++v) {
            sum += std::exp((double) (z[v] - zmax));
        }

        // gradient ascent on w * log softmax(W x)[token], then the pull toward W0
        for (int64_t v = 0; v < n_vocab; ++v) {
            const double prob = std::exp((double) (z[v] - zmax)) / sum;
            const float  c    = (float) (lr * s.weight * ((s.token == v ? 1.0 : 0.0) - prob));
            float * wv = W + v * R;
            if (reg > 0.0f) {
                const float * w0v = &w0[v * R];
                const float shrink = lr * reg;
                for (int64_t r = 0; r < R; ++r) {
                    wv[r] -= shrink * (wv[r] - w0v[r]);
                }
            }
            esn_axpy(wv, c, x, R);
        }
    }
}

void llama_esn_adapter::reset() {
    pending.clear();
    pending_events = 0;
    history.clear();
    rls_p.clear();
    rls_p.shrink_to_fit();
    last_mode = -1;
    active    = false;
    // the readout buffer is kept: graphs may still reference the tensor, and
    // the next update rewrites it from the checkpoint
    if (w != nullptr) {
        memcpy(w_data(), w0.data(), w0.size() * sizeof(float));
    }
}

//
// API
//

#include "llama-context.h"

llama_esn_online_params llama_esn_online_params_default(void) {
    llama_esn_online_params p = {
        /*.enabled          =*/ false,
        /*.update_mode      =*/ 0,
        /*.learning_rate    =*/ 1e-4f,
        /*.regularization   =*/ 1e-6f,
        /*.decay_rate       =*/ 0.0f,
        /*.replay_window    =*/ 256,
        /*.freeze_reservoir =*/ true,
    };
    return p;
}

llama_esn_online_params llama_esn_model_online_params(const llama_model * model) {
    llama_esn_online_params p = llama_esn_online_params_default();
    if (model == nullptr || model->arch != LLM_ARCH_ESN) {
        return p;
    }
    const auto & hp = model->hparams;
    p.enabled          = hp.esn_online_learning;
    p.update_mode      = hp.esn_online_update_mode;
    p.learning_rate    = hp.esn_online_lr;
    p.regularization   = hp.esn_online_reg;
    p.decay_rate       = hp.esn_online_decay_rate;
    p.replay_window    = hp.esn_replay_window;
    p.freeze_reservoir = hp.esn_freeze_reservoir;
    return p;
}

bool llama_esn_is_adaptive(const llama_model * model) {
    return model != nullptr && model->arch == LLM_ARCH_ESN && model->hparams.esn_online_learning;
}

uint32_t llama_esn_reservoir_size(const llama_model * model) {
    return model != nullptr && model->arch == LLM_ARCH_ESN ? model->hparams.esn_reservoir_size : 0;
}

uint32_t llama_esn_n_levels(const llama_model * model) {
    return model != nullptr && model->arch == LLM_ARCH_ESN ? model->hparams.esn_n_levels : 0;
}

// -1 not an ESN, -4 ESN without online learning, 0 adapter available
static int32_t llama_esn_check(const llama_context * ctx) {
    const llama_model & model = ctx->get_model();
    if (model.arch != LLM_ARCH_ESN) {
        return -1;
    }
    if (ctx->get_esn() == nullptr) {
        LLAMA_LOG_ERROR("%s: the model was not exported with esn.online_learning.enabled\n", __func__);
        return -4;
    }
    return 0;
}

int32_t llama_esn_submit_feedback(
        llama_context *                  ctx,
        const llama_esn_feedback_event * events,
        int32_t                          n_events) {
    if (ctx == nullptr) {
        return -2;
    }
    if (const int32_t rc = llama_esn_check(ctx); rc != 0) {
        return rc;
    }
    const llama_memory_t mem = ctx->get_memory();
    return ctx->get_esn()->submit(events, n_events, [mem](llama_seq_id seq) {
        return mem ? mem->seq_pos_max(seq) : (llama_pos) -1;
    });
}

int32_t llama_esn_update_readout(
        llama_context *                 ctx,
        const llama_esn_online_params * params) {
    if (ctx == nullptr) {
        return -2;
    }
    if (const int32_t rc = llama_esn_check(ctx); rc != 0) {
        return rc;
    }
    if (params == nullptr) {
        return -3;
    }
    if (params->enabled && ctx->lora_adapts_readout()) {
        LLAMA_LOG_ERROR("%s: a LoRA adapter on this context modifies the readout; the adapted readout would "
                "replace it, so remove the adapter (llama_rm_adapter_lora) before adapting\n", __func__);
        return -10;
    }
    // no graph may be reading the readout while it changes
    ctx->synchronize();
    try {
        return ctx->get_esn()->update(*params, (int) ctx->n_threads_batch());
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return -8;
    }
}

void llama_esn_reset_adaptation(llama_context * ctx) {
    if (ctx == nullptr || ctx->get_esn() == nullptr) {
        return;
    }
    ctx->synchronize();
    ctx->get_esn()->reset();
}
