#include "llama-hparams.h"

#include "ggml.h"
#include <cassert>
#include <stdexcept>
#include <string>

void llama_hparams::set_swa_pattern(uint32_t n_pattern, bool dense_first) {
    if (dense_first) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            swa_layers[il] = n_pattern == 0 || (il % n_pattern != 0);
        }
    } else {
        for (uint32_t il = 0; il < n_layer; ++il) {
            swa_layers[il] = n_pattern == 0 || (il % n_pattern < (n_pattern - 1));
        }
    }
}

bool llama_hparams::is_swa_any() const {
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (swa_layers[il]) {
            return true;
        }
    }

    return false;
}

uint32_t llama_hparams::n_head(uint32_t il) const {
    if (il < n_layer) {
        return n_head_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_head_kv(uint32_t il) const {
    if (il < n_layer) {
        return n_head_kv_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_ff(uint32_t il) const {
    if (il < n_layer) {
        return n_ff_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_gqa(uint32_t il) const {
    const uint32_t n_head    = this->n_head(il);
    const uint32_t n_head_kv = this->n_head_kv(il);

    if (n_head_kv == 0) {
        return 0;
    }

    return n_head/n_head_kv;
}

uint32_t llama_hparams::n_embd_k_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_k * n_head_kv;
}

uint32_t llama_hparams::n_embd_v_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_v * n_head_kv;
}

bool llama_hparams::is_n_embd_k_gqa_variable() const {
    const uint32_t val = n_embd_k_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (val != n_embd_k_gqa(il)) {
            return true;
        }
    }

    return false;
}

bool llama_hparams::is_n_embd_v_gqa_variable() const {
    const uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (val != n_embd_v_gqa(il)) {
            return true;
        }
    }

    return false;
}

uint32_t llama_hparams::n_embd_k_gqa_max() const {
    uint32_t val = n_embd_k_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        val = std::max(val, n_embd_k_gqa(il));
    }

    return val;
}

uint32_t llama_hparams::n_embd_v_gqa_max() const {
    uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        val = std::max(val, n_embd_v_gqa(il));
    }

    return val;
}

uint32_t llama_hparams::n_embd_r() const {
    if (wkv_head_size != 0) {
        // for RWKV models
        return token_shift_count * n_embd;
    }

    if (n_shortconv_l_cache != 0) {
        // for LFM2 models
        return n_embd * (n_shortconv_l_cache - 1);
    }

    // TODO: maybe support other convolution strides than 1
    // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
    // Corresponds to Mamba's conv_states size
    return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state);
}

uint32_t llama_hparams::n_embd_s() const {
    if (esn_reservoir_size != 0) {
        // corresponds to ESN's reservoir state size
        return esn_reservoir_size;
    }

    if (wkv_head_size != 0) {
        // corresponds to RWKV's wkv_states size
        return n_embd * wkv_head_size;
    }

    // corresponds to Mamba's ssm_states size
    return ssm_d_state * ssm_d_inner;
}

bool llama_hparams::is_recurrent(uint32_t il) const {
    return recurrent_layer_arr[il];
}

uint32_t llama_hparams::n_pos_per_embd() const {
    return rope_type == LLAMA_ROPE_TYPE_MROPE ? 4 : 1;
}

bool llama_hparams::is_swa(uint32_t il) const {
    if (il < n_layer) {
        return swa_layers[il];
    }

    GGML_ABORT("fatal error");
}

bool llama_hparams::has_kv(uint32_t il) const {
    if (n_layer_kv_from_start >= 0) {
        if (il < (uint32_t) n_layer_kv_from_start) {
            return true;
        }

        return false;
    }

    // by default, all layers have kv
    return true;
}

uint32_t llama_hparams::n_layer_kv() const {
    uint32_t res = 0;

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (has_kv(il)) {
            res++;
        }
    }

    return res;
}

bool llama_hparams::is_masked_swa(uint32_t n_swa, llama_swa_type swa_type, llama_pos p0, llama_pos p1) {
    assert(p0 >= 0 && p1 >= 0);

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:
            {
            } break;
        case LLAMA_SWA_TYPE_STANDARD:
            {
                if (p1 - p0 >= (int32_t) n_swa) {
                    return true;
                }
            } break;
        case LLAMA_SWA_TYPE_CHUNKED:
            {
                const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                if (p0 < pos_chunk_start) {
                    return true;
                }
            } break;
        case LLAMA_SWA_TYPE_SYMMETRIC:
            {
                const int32_t half_n_swa = (int32_t) n_swa / 2;
                const int32_t pos_diff = p1 - p0;

                // Mask if outside the symmetric window
                if (pos_diff < -half_n_swa || pos_diff > half_n_swa) {
                    return true;
                }
            } break;
    }

    return false;
}

void llama_hparams::validate_esn_params() const {
    // reservoir_size must be set to a positive value for a valid ESN model
    if (esn_reservoir_size == 0) {
        throw std::invalid_argument("ESN: esn_reservoir_size must be > 0");
    }

    // spectral_radius must be in (0, 1) to guarantee the echo state property
    if (esn_spectral_radius <= 0.0f || esn_spectral_radius >= 1.0f) {
        throw std::invalid_argument(
            "ESN: esn_spectral_radius must be in (0, 1), got " +
            std::to_string(esn_spectral_radius));
    }

    // sparsity must be in [0, 1]
    if (esn_sparsity < 0.0f || esn_sparsity > 1.0f) {
        throw std::invalid_argument(
            "ESN: esn_sparsity must be in [0, 1], got " +
            std::to_string(esn_sparsity));
    }

    // leaking_rate must be in (0, 1]
    if (esn_leaking_rate <= 0.0f || esn_leaking_rate > 1.0f) {
        throw std::invalid_argument(
            "ESN: esn_leaking_rate must be in (0, 1], got " +
            std::to_string(esn_leaking_rate));
    }

    // input_scaling must be positive
    if (esn_input_scaling <= 0.0f) {
        throw std::invalid_argument(
            "ESN: esn_input_scaling must be > 0, got " +
            std::to_string(esn_input_scaling));
    }

    // feedback_scaling must be non-negative
    if (esn_feedback_scaling < 0.0f) {
        throw std::invalid_argument(
            "ESN: esn_feedback_scaling must be >= 0, got " +
            std::to_string(esn_feedback_scaling));
    }

    // noise_level must be non-negative
    if (esn_noise_level < 0.0f) {
        throw std::invalid_argument(
            "ESN: esn_noise_level must be >= 0, got " +
            std::to_string(esn_noise_level));
    }

    // activation_type: 0=tanh, 1=sigmoid, 2=leaky_relu
    if (esn_activation_type > 2) {
        throw std::invalid_argument(
            "ESN: esn_activation_type must be 0 (tanh), 1 (sigmoid), or 2 (leaky_relu), got " +
            std::to_string(esn_activation_type));
    }

    // Online learning: learning rate and regularization must be positive when enabled
    if (esn_online_learning) {
        if (esn_online_lr <= 0.0f) {
            throw std::invalid_argument(
                "ESN: esn_online_lr must be > 0 when online learning is enabled, got " +
                std::to_string(esn_online_lr));
        }
        if (esn_online_reg < 0.0f) {
            throw std::invalid_argument(
                "ESN: esn_online_reg must be >= 0, got " +
                std::to_string(esn_online_reg));
        }
        if (esn_online_update_mode > 2) {
            throw std::invalid_argument(
                "ESN: esn_online_update_mode must be 0 (batch_ridge), 1 (rls), or 2 (sgd), got " +
                std::to_string(esn_online_update_mode));
        }
        if (esn_online_decay_rate < 0.0f || esn_online_decay_rate >= 1.0f) {
            throw std::invalid_argument(
                "ESN: esn_online_decay_rate must be in [0, 1), got " +
                std::to_string(esn_online_decay_rate));
        }
    }

    // Hierarchical parameters
    if (esn_n_levels == 0) {
        throw std::invalid_argument("ESN: esn_n_levels must be >= 1");
    }
    if (esn_n_levels > 1 && esn_branching_factor == 0) {
        throw std::invalid_argument("ESN: esn_branching_factor must be >= 1 for hierarchical ESN (n_levels > 1)");
    }
}
