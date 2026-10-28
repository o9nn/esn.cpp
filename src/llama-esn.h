#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <deque>
#include <map>
#include <vector>

struct llama_model;
struct llama_ubatch;

// Online readout adaptation for an ESN context (the llama_esn_* API).
//
// The reservoir is fixed; only the linear readout W_out is trained. The
// context records the readout input x (result_norm, one vector per output
// token) after every ubatch, keyed by (seq_id, pos). Feedback events turn
// those recorded vectors into training samples (x, target token, weight), and
// llama_esn_update_readout() fits a per-context copy of W_out to them, which
// the graph then uses in place of the model's tensor. The model's own W_out
// (the checkpoint W0) is never modified, so contexts adapt independently.
//
// Objectives, with y the one-hot target and T = esn.output_temperature (the
// training script fits W_out to T * y and keeps that scale):
//   0 ridge: W = argmin  sum_i w_i |T y_i - W x_i|^2 + lambda |W - W0|^2
//            over the newest replay_window samples, solved from W0 each time
//            (Woodbury form, an n x n system for n samples)
//   1 RLS:   the same objective solved recursively from the current W, with
//            forgetting factor 1 - decay; P = (lambda I + sum w x x^T)^-1
//   2 SGD:   one cross-entropy step per sample on softmax(W x), weighted by w
//            (negative w pushes probability away from the token), plus an L2
//            pull toward W0
struct llama_esn_adapter {
    explicit llama_esn_adapter(const llama_model & model);
    ~llama_esn_adapter();

    // Record the readout inputs of one ubatch. `rows` holds n_outputs
    // vectors of reservoir_size floats, in ubatch output order.
    void capture(const llama_ubatch & ubatch, uint32_t n_outputs, const float * rows);

    int32_t submit (const llama_esn_feedback_event * events, int32_t n_events);
    int32_t update (const llama_esn_online_params & params, int n_threads);
    void    reset();

    // the adapted readout, or nullptr while the checkpoint is in use
    ggml_tensor * readout() const { return active ? w : nullptr; }

    size_t n_pending_samples() const { return pending.size(); }

private:
    struct captured {
        llama_pos          pos;
        std::vector<float> x;
    };

    struct sample {
        std::vector<float> x;
        llama_token        token;
        float              weight;
    };

    const llama_model & model;

    const int64_t n_res;   // reservoir size: readout input width
    const int64_t n_vocab; // readout output width

    // recorded readout inputs per sequence, oldest first, ascending pos
    std::map<llama_seq_id, std::deque<captured>> recorded;
    size_t recorded_cap; // per sequence

    std::vector<sample> pending; // submitted, not yet trained on
    int32_t pending_events = 0;
    size_t  pending_cap;

    std::deque<sample> history;  // trained-on samples, newest last

    // checkpoint readout as F32, [n_vocab][n_res]; loaded on first update
    std::vector<float> w0;

    // adapted readout, F32 [n_res, n_vocab] in a host buffer
    ggml_context_ptr        ctx_w;
    ggml_backend_buffer_ptr buf_w;
    ggml_tensor *           w      = nullptr;
    bool                    active = false;

    // RLS inverse correlation matrix, [n_res][n_res], valid for rls_lambda
    std::vector<double> rls_p;
    float               rls_lambda = 0.0f;
    int                 last_mode  = -1;

    const captured * find(llama_seq_id seq, llama_pos pos) const;

    void   ensure_readout();
    float * w_data() const;

    void update_ridge(float lambda, float decay, uint32_t window, float temperature, int n_threads);
    void update_rls  (float lambda, float decay, float temperature);
    void update_sgd  (float lr, float reg);
};
