// End-to-end ESN inference smoke test.
//
// Reads a GGUF produced by scripts/esn_training.py (passed on the command
// line) and runs a single decode step through the public llama C API with
// raw token IDs. This exercises:
//   - GGUF metadata and tensor shape loading for arch=esn
//   - llama_memory_recurrent allocation (n_embd_s() -> reservoir_size)
//   - llm_build_esn forward graph construction and execution
//   - logits readout
//
// The test is "smoke" only: it verifies that llama_decode returns success
// and that the logits are finite and have the right dimensionality. It does
// not score accuracy, since the model may be untrained.
//
// Invocation (from the build dir):
//   bin/test-esn-inference <path-to-esn-model.gguf>
//
// To regenerate a model for this test, run (as a single line):
//   python scripts/esn_training.py --init-only --reservoir-size 128
//     --vocab-size 1024 --embedding-dim 64 --output /tmp/esn-smoke.gguf

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void silent_log(enum ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <esn.gguf>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string model_path = argv[1];
    fprintf(stderr, "test-esn-inference: using '%s'\n", model_path.c_str());

    // Keep stderr tidy during the test.
    llama_log_set(silent_log, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = false;
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "FAIL: could not load model\n");
        return EXIT_FAILURE;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    assert(n_vocab > 0);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 32;
    cparams.n_batch     = 32;
    cparams.n_ubatch    = 32;
    cparams.n_seq_max   = 1;
    cparams.no_perf     = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "FAIL: could not create context\n");
        llama_model_free(model);
        return EXIT_FAILURE;
    }

    // Feed tokens one at a time — this is the natural recurrent-inference
    // pattern. The ESN builder processes one update per sequence per decode
    // call, which matches a single-token batch exactly.
    const std::vector<llama_token> input_tokens = { 1, 2, 3, 4 };
    for (auto id : input_tokens) {
        llama_token t = (llama_token) ((int32_t) id % n_vocab);
        llama_batch batch = llama_batch_get_one(&t, 1);
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL: llama_decode returned %d\n", rc);
            llama_free(ctx);
            llama_model_free(model);
            return EXIT_FAILURE;
        }
    }

    // Read logits for the most recent decode and sanity-check.
    const float * logits = llama_get_logits_ith(ctx, 0);
    if (logits == nullptr) {
        fprintf(stderr, "FAIL: llama_get_logits_ith returned null\n");
        llama_free(ctx);
        llama_model_free(model);
        return EXIT_FAILURE;
    }

    int n_finite = 0;
    int n_nonzero = 0;
    for (int i = 0; i < n_vocab; ++i) {
        if (std::isfinite(logits[i])) {
            ++n_finite;
        }
        if (logits[i] != 0.0f) {
            ++n_nonzero;
        }
    }

    assert(n_finite == n_vocab && "all logits must be finite");
    // We can't assert that logits are non-zero because an untrained model
    // might legitimately produce zeros (W_out initialized to zero), but
    // we at least want to see the computation ran.
    fprintf(stderr,
            "test-esn-inference: decoded %zu tokens, got %d-dim logits (%d finite, %d nonzero)\n",
            input_tokens.size(), n_vocab, n_finite, n_nonzero);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "test-esn-inference: PASSED\n");
    return EXIT_SUCCESS;
}
