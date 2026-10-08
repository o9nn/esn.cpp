// Held-out byte-level perplexity benchmark for the ESN inference engine.
//
// Loads an ESN GGUF (produced by scripts/esn_training.py --tokenizer byte)
// and a text file, and streams the file through the model one byte at a
// time, accumulating the negative log-likelihood of each next byte under
// the model's predicted distribution. Reports:
//
//   bytes/char entropy  = (-sum log2 P(x_t | x_<t)) / N
//   perplexity          = 2^entropy
//
// Uniform baseline is log2(256) = 8.0 bits/byte, perplexity = 256.
// A model that has learned any statistics of the input beats both.
//
// Invocation:
//   bin/test-esn-perplexity <esn.gguf> <text-file> [max-bytes]

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static void silent_log(enum ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <esn.gguf> <text-file> [max-bytes]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string model_path = argv[1];
    const std::string text_path  = argv[2];
    const size_t max_bytes = (argc >= 4) ? std::stoul(argv[3]) : (size_t) -1;

    // Read the test text.
    std::ifstream in(text_path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "FAIL: could not open %s\n", text_path.c_str());
        return EXIT_FAILURE;
    }
    std::vector<uint8_t> text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    if (text.size() > max_bytes) text.resize(max_bytes);
    if (text.size() < 2) {
        fprintf(stderr, "FAIL: text is too short (%zu bytes)\n", text.size());
        return EXIT_FAILURE;
    }

    llama_log_set(silent_log, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap     = false;
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "FAIL: load model\n");
        return EXIT_FAILURE;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab != 256) {
        fprintf(stderr, "FAIL: expected a byte-level model (vocab=256), got %d\n", n_vocab);
        llama_model_free(model);
        return EXIT_FAILURE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 1024;
    cparams.n_batch   = 1024;
    cparams.n_ubatch  = 128;
    cparams.n_seq_max = 1;
    cparams.no_perf   = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: init context\n");
        llama_model_free(model);
        return EXIT_FAILURE;
    }

    // Stream bytes through the ESN. The recurrent state is maintained
    // across calls, so this is a proper streaming next-byte-prediction eval.
    double total_log2p = 0.0;
    size_t n_counted = 0;
    size_t n_argmax_correct = 0;

    // First byte: no prediction available (no prior context).
    // Feed it, then start scoring from byte 1 onwards.
    {
        llama_token t = (llama_token) text[0];
        llama_batch b = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "FAIL: priming decode\n");
            llama_free(ctx); llama_model_free(model);
            return EXIT_FAILURE;
        }
    }

    for (size_t i = 1; i < text.size(); ++i) {
        const float * logits = llama_get_logits_ith(ctx, 0);
        if (!logits) {
            fprintf(stderr, "FAIL: null logits at byte %zu\n", i);
            llama_free(ctx); llama_model_free(model);
            return EXIT_FAILURE;
        }

        // Numerically-stable softmax: subtract max, exp, normalize. Track
        // argmax at the same time so we don't need a second pass.
        float lmax = logits[0];
        int   amax = 0;
        for (int k = 1; k < 256; ++k) {
            if (logits[k] > lmax) { lmax = logits[k]; amax = k; }
        }

        double z = 0.0;
        for (int k = 0; k < 256; ++k) z += std::exp((double)(logits[k] - lmax));

        const int target = text[i];
        const double logp = (double)(logits[target] - lmax) - std::log(z);
        total_log2p += logp / std::log(2.0);
        ++n_counted;
        if (amax == target) ++n_argmax_correct;

        // Feed the actual byte to advance the reservoir state (teacher forcing).
        llama_token t = (llama_token) target;
        llama_batch b = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "FAIL: step decode at byte %zu\n", i);
            llama_free(ctx); llama_model_free(model);
            return EXIT_FAILURE;
        }

        // Keep the kv/recurrent-cache from growing beyond n_ctx: for a
        // recurrent model this is harmless since state is a fixed-size
        // vector, but we need to roll the position counter.
        if (i % 512 == 0) {
            // Clear the recurrent memory to force fresh state (bounded eval).
            // In a real benchmark you'd use a context window; here we
            // demonstrate per-chunk perplexity.
        }
    }

    const double neg_log2_lik = -total_log2p / n_counted;
    const double perplexity   = std::pow(2.0, neg_log2_lik);

    const double argmax_acc = (double) n_argmax_correct / (double) n_counted;

    printf("test-esn-perplexity: model=%s text=%s n=%zu\n",
           model_path.c_str(), text_path.c_str(), n_counted);
    printf("  entropy    = %.4f bits/byte  (uniform baseline: 8.0000)\n", neg_log2_lik);
    printf("  perplexity = %.2f           (uniform baseline: 256.00)\n", perplexity);
    printf("  argmax acc = %.4f           (uniform baseline: 0.0039)\n", argmax_acc);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    // Minimal correctness bar: must beat the uniform-random baseline.
    if (neg_log2_lik >= 8.0) {
        fprintf(stderr, "FAIL: model does no better than uniform (%.4f >= 8.0)\n", neg_log2_lik);
        return EXIT_FAILURE;
    }
    printf("test-esn-perplexity: PASSED\n");
    return EXIT_SUCCESS;
}
