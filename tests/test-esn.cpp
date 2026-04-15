#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-model.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

// Helper: build a minimally valid ESN hparams with reservoir_size set
static llama_hparams make_valid_esn_hparams(uint32_t rsz = 512) {
    llama_hparams h;
    h.esn_reservoir_size  = rsz;
    h.esn_spectral_radius = 0.9f;
    h.esn_sparsity        = 0.1f;
    h.esn_leaking_rate    = 0.5f;
    h.esn_input_scaling   = 1.0f;
    return h;
}

int main() {
    std::cout << "Testing ESN architecture support...\n\n";

    // ------------------------------------------------------------------ //
    // Test 1: Architecture recognition and name mapping
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 1: ESN architecture recognition... ";
        llm_arch esn_arch = llm_arch_from_string("esn");
        assert(esn_arch == LLM_ARCH_ESN);
        assert(std::string(llm_arch_name(esn_arch)) == "esn");
        // Unknown arch must not collide with ESN
        assert(llm_arch_from_string("not_an_arch") == LLM_ARCH_UNKNOWN);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 2: ESN is recurrent, not hybrid, not diffusion
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 2: ESN recurrent / non-hybrid / non-diffusion... ";
        assert(llm_arch_is_recurrent(LLM_ARCH_ESN)  == true);
        assert(llm_arch_is_hybrid(LLM_ARCH_ESN)     == false);
        assert(llm_arch_is_diffusion(LLM_ARCH_ESN)  == false);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 3: ESN tensor info mappings
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 3: ESN tensor info mappings... ";
        const llm_tensor_info & input_info = llm_tensor_info_for(LLM_TENSOR_ESN_INPUT_WEIGHTS);
        assert(input_info.layer == LLM_TENSOR_LAYER_INPUT);
        assert(input_info.op   == GGML_OP_MUL_MAT);

        const llm_tensor_info & reservoir_info = llm_tensor_info_for(LLM_TENSOR_ESN_RESERVOIR_WEIGHTS);
        assert(reservoir_info.layer == LLM_TENSOR_LAYER_REPEATING);
        assert(reservoir_info.op   == GGML_OP_MUL_MAT);

        const llm_tensor_info & output_info = llm_tensor_info_for(LLM_TENSOR_ESN_OUTPUT_WEIGHTS);
        assert(output_info.layer == LLM_TENSOR_LAYER_OUTPUT);
        assert(output_info.op   == GGML_OP_MUL_MAT);

        // Optional tensor infos
        const llm_tensor_info & fb_info  = llm_tensor_info_for(LLM_TENSOR_ESN_FEEDBACK_WEIGHTS);
        assert(fb_info.op == GGML_OP_MUL_MAT);
        const llm_tensor_info & ib_info  = llm_tensor_info_for(LLM_TENSOR_ESN_INPUT_BIAS);
        assert(ib_info.op == GGML_OP_ADD);
        const llm_tensor_info & rb_info  = llm_tensor_info_for(LLM_TENSOR_ESN_RESERVOIR_BIAS);
        assert(rb_info.op == GGML_OP_ADD);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 4: Default hyperparameter values
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 4: ESN hyperparameter defaults... ";
        llama_hparams h;
        assert(h.esn_reservoir_size   == 0);
        assert(h.esn_spectral_radius  == 0.95f);
        assert(h.esn_sparsity         == 0.1f);
        assert(h.esn_leaking_rate     == 1.0f);
        assert(h.esn_input_scaling    == 1.0f);
        assert(h.esn_feedback_scaling == 0.0f);
        assert(h.esn_noise_level      == 0.0f);
        assert(h.esn_activation_type  == 0);
        assert(h.esn_bidirectional    == false);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 5: Online learning defaults
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 5: Online learning hyperparameter defaults... ";
        llama_hparams h;
        assert(h.esn_online_learning    == false);
        assert(h.esn_freeze_reservoir   == true);
        assert(h.esn_online_update_mode == 0);
        assert(h.esn_online_buffer_size == 64);
        assert(h.esn_replay_window      == 256);
        assert(h.esn_online_decay_rate  == 0.0f);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 6: Hierarchical ESN defaults
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 6: Hierarchical ESN hyperparameter defaults... ";
        llama_hparams h;
        assert(h.esn_n_levels         == 1);
        assert(h.esn_branching_factor == 3);
        assert(h.esn_lateral_links    == false);
        assert(h.esn_top_down_mod     == false);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 7: Tensor name generation
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 7: ESN tensor name generation... ";
        const auto tn = LLM_TN(LLM_ARCH_ESN);

        // Assign to std::string first (LLM_TN_IMPL has implicit string conversion)
        std::string input_name     = tn(LLM_TENSOR_ESN_INPUT_WEIGHTS);
        std::string reservoir_name = tn(LLM_TENSOR_ESN_RESERVOIR_WEIGHTS);
        std::string output_name    = tn(LLM_TENSOR_ESN_OUTPUT_WEIGHTS);
        std::string feedback_name  = tn(LLM_TENSOR_ESN_FEEDBACK_WEIGHTS);
        std::string input_bias     = tn(LLM_TENSOR_ESN_INPUT_BIAS);
        std::string reservoir_bias = tn(LLM_TENSOR_ESN_RESERVOIR_BIAS);

        assert(input_name     == "esn_input_weights");
        assert(reservoir_name == "esn_reservoir_weights");
        assert(output_name    == "esn_output_weights");
        assert(feedback_name  == "esn_feedback_weights");
        assert(input_bias     == "esn_input_bias");
        assert(reservoir_bias == "esn_reservoir_bias");
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 8: n_embd_s() returns reservoir size for ESN models
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 8: n_embd_s() returns reservoir size for ESN... ";
        llama_hparams h = make_valid_esn_hparams(1024);
        assert(h.n_embd_s() == 1024);

        // When reservoir_size == 0 the function falls through to the SSM path.
        // With all SSM fields at default (0), n_embd_s() == ssm_d_state * ssm_d_inner == 0.
        llama_hparams h2;
        h2.esn_reservoir_size = 0;
        assert(h2.n_embd_s() == 0);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 9: Parameter validation – valid configuration passes
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 9: validate_esn_params() – valid config... ";
        llama_hparams h = make_valid_esn_hparams();
        // Must not throw
        try {
            h.validate_esn_params();
        } catch (const std::invalid_argument & e) {
            std::cerr << "\nUnexpected exception: " << e.what() << "\n";
            return EXIT_FAILURE;
        }
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 10: Validation rejects reservoir_size == 0
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 10: validate_esn_params() – zero reservoir_size rejected... ";
        llama_hparams h = make_valid_esn_hparams();
        h.esn_reservoir_size = 0;
        bool threw = false;
        try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 11: Validation rejects spectral_radius >= 1
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 11: validate_esn_params() – spectral_radius >= 1 rejected... ";
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_spectral_radius = 1.0f;
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_spectral_radius = 1.5f;
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_spectral_radius = 0.0f;
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 12: Validation rejects leaking_rate out of (0, 1]
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 12: validate_esn_params() – leaking_rate bounds... ";
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_leaking_rate = 0.0f;   // exclusive lower bound
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_leaking_rate = 1.5f;   // above 1
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_leaking_rate = 1.0f;   // inclusive upper bound – OK
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(!threw);
        }
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 13: Validation rejects invalid activation_type
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 13: validate_esn_params() – activation_type bounds... ";
        for (uint32_t valid : {0u, 1u, 2u}) {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_activation_type = valid;
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(!threw);
        }
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_activation_type = 3;   // invalid
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 14: Validation rejects non-positive input_scaling
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 14: validate_esn_params() – non-positive input_scaling rejected... ";
        // Negative value
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_input_scaling = -0.1f;
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        // Zero is also invalid (must be > 0)
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_input_scaling = 0.0f;
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 15: Online learning validation – bad lr rejected when enabled
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 15: validate_esn_params() – online lr=0 rejected when enabled... ";
        llama_hparams h = make_valid_esn_hparams();
        h.esn_online_learning = true;
        h.esn_online_lr       = 0.0f;   // must be > 0
        bool threw = false;
        try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 16: Online learning validation – bad update_mode rejected
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 16: validate_esn_params() – invalid online update_mode rejected... ";
        llama_hparams h = make_valid_esn_hparams();
        h.esn_online_learning    = true;
        h.esn_online_lr          = 1e-3f;
        h.esn_online_update_mode = 5;    // only 0,1,2 valid
        bool threw = false;
        try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 17: Online learning validation passes with valid params
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 17: validate_esn_params() – valid online learning config... ";
        for (uint32_t mode : {0u, 1u, 2u}) {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_online_learning    = true;
            h.esn_online_lr          = 1e-4f;
            h.esn_online_reg         = 1e-6f;
            h.esn_online_update_mode = mode;
            h.esn_online_decay_rate  = 0.0f;
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(!threw);
        }
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 18: Hierarchical validation – n_levels == 0 rejected
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 18: validate_esn_params() – n_levels == 0 rejected... ";
        llama_hparams h = make_valid_esn_hparams();
        h.esn_n_levels = 0;
        bool threw = false;
        try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 19: Hierarchical validation – branching_factor == 0 with n_levels > 1 rejected
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 19: validate_esn_params() – branching_factor == 0 with n_levels > 1 rejected... ";
        llama_hparams h = make_valid_esn_hparams();
        h.esn_n_levels         = 3;
        h.esn_branching_factor = 0;  // invalid for hierarchical
        bool threw = false;
        try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 20: Online decay_rate == 1 is invalid (open interval [0, 1))
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 20: validate_esn_params() – decay_rate >= 1 rejected when online enabled... ";
        llama_hparams h = make_valid_esn_hparams();
        h.esn_online_learning   = true;
        h.esn_online_lr         = 1e-3f;
        h.esn_online_decay_rate = 1.0f;   // must be < 1
        bool threw = false;
        try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 21: Negative noise_level rejected
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 21: validate_esn_params() – negative noise_level rejected... ";
        llama_hparams h = make_valid_esn_hparams();
        h.esn_noise_level = -0.01f;
        bool threw = false;
        try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 22: Negative feedback_scaling rejected
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 22: validate_esn_params() – negative feedback_scaling rejected... ";
        llama_hparams h = make_valid_esn_hparams();
        h.esn_feedback_scaling = -0.5f;
        bool threw = false;
        try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
        std::cout << "PASSED\n";
    }

    // ------------------------------------------------------------------ //
    // Test 23: sparsity outside [0,1] rejected
    // ------------------------------------------------------------------ //
    {
        std::cout << "Test 23: validate_esn_params() – sparsity outside [0,1] rejected... ";
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_sparsity = -0.1f;
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        {
            llama_hparams h = make_valid_esn_hparams();
            h.esn_sparsity = 1.1f;
            bool threw = false;
            try { h.validate_esn_params(); } catch (const std::invalid_argument &) { threw = true; }
            assert(threw);
        }
        std::cout << "PASSED\n";
    }

    std::cout << "\nAll " << 23 << " ESN tests passed successfully!\n";

    // Summary of ESN architecture capabilities
    std::cout << "\nESN Architecture Features:\n";
    std::cout << "  Recurrent:        " << (llm_arch_is_recurrent(LLM_ARCH_ESN)  ? "Yes" : "No") << "\n";
    std::cout << "  Hybrid:           " << (llm_arch_is_hybrid(LLM_ARCH_ESN)     ? "Yes" : "No") << "\n";
    std::cout << "  Diffusion:        " << (llm_arch_is_diffusion(LLM_ARCH_ESN)  ? "Yes" : "No") << "\n";
    std::cout << "  Architecture name: " << llm_arch_name(LLM_ARCH_ESN)                          << "\n";

    return EXIT_SUCCESS;
}
