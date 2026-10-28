// Accuracy of the ESN reservoir activations on the CPU backend.
//
// ggml-cpu computes tanh and sigmoid branch-free (no libm call) so the loops
// vectorize; the reservoir applies one of them to every state element at
// every timestep. This checks them against double-precision references:
//
//   A. tanh within 2 ulp over [-12, 12], and exact at the special values
//      (±0, ±inf, NaN, tiny inputs).
//   B. sigmoid within 4 ulp over [-80, 80], saturating at ±inf, NaN kept.
//   C. The same results with several threads and rows, so the vectorized
//      body and the scalar tail of each row agree.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Distance in units in the last place of `want` (rounded to float).
static double ulp_err(float got, double want) {
    const float w = (float) want;
    if (std::isnan(w)) {
        return std::isnan(got) ? 0.0 : INFINITY;
    }
    if (std::isinf(w)) {
        return got == w ? 0.0 : INFINITY;
    }
    const float up = std::nextafter(std::fabs(w), INFINITY) - std::fabs(w);
    return std::fabs((double) got - want) / up;
}

enum class act { tanh, sigmoid };

// Apply the activation with ggml to x laid out as [ncols, nrows].
static std::vector<float> run(act a, const std::vector<float> & x, int64_t ncols, int n_threads) {
    const int64_t nrows = (int64_t) x.size() / ncols;
    ggml_init_params ip = { /*mem_size*/ 2 * x.size() * sizeof(float) + 4 * ggml_tensor_overhead() + ggml_graph_overhead(),
                            /*mem_buffer*/ nullptr, /*no_alloc*/ false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    memcpy(in->data, x.data(), x.size() * sizeof(float));
    ggml_tensor * out = a == act::tanh ? ggml_tanh(ctx, in) : ggml_sigmoid(ctx, in);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    assert(st == GGML_STATUS_SUCCESS);
    std::vector<float> y((const float *) out->data, (const float *) out->data + x.size());
    ggml_free(ctx);
    return y;
}

// Evenly spaced bit patterns of the non-negative floats up to `hi`, mirrored.
static std::vector<float> sweep(float hi, int n) {
    uint32_t top;
    memcpy(&top, &hi, sizeof(top));
    std::vector<float> x;
    for (int i = 0; i < n; ++i) {
        uint32_t b = (uint32_t) ((uint64_t) top * i / (n - 1));
        float v;
        memcpy(&v, &b, sizeof(v));
        x.push_back(v);
        x.push_back(-v);
    }
    return x;
}

static double ref(act a, float x) {
    return a == act::tanh ? std::tanh((double) x) : 1.0 / (1.0 + std::exp(-(double) x));
}

static void check(act a, const char * name, float hi, double max_ulp) {
    const std::vector<float> x = sweep(hi, 1 << 18);
    const std::vector<float> y = run(a, x, (int64_t) x.size(), 1);
    double worst = 0.0;
    float worst_x = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        const double e = ulp_err(y[i], ref(a, x[i]));
        if (e > worst) {
            worst = e;
            worst_x = x[i];
        }
    }
    printf("  %-7s max error %.2f ulp (at x = %g, limit %.0f)\n", name, worst, worst_x, max_ulp);
    assert(worst <= max_ulp);

    // C: several rows with an odd length, so each row has a scalar tail, and
    // several threads; every element must match the single-row result.
    const int64_t ncols = 1027;
    const std::vector<float> xs(x.begin(), x.begin() + ncols * 64);
    const std::vector<float> one  = run(a, xs, (int64_t) xs.size(), 1);
    const std::vector<float> rows = run(a, xs, ncols, 4);
    assert(memcmp(one.data(), rows.data(), one.size() * sizeof(float)) == 0);
}

int main() {
    printf("test-esn-activations:\n");

    check(act::tanh, "tanh", 12.0f, 2.0);
    {
        const std::vector<float> x = { 0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1e-30f, -1e-30f, 1e-40f, 20.0f, -20.0f };
        const std::vector<float> y = run(act::tanh, x, (int64_t) x.size(), 1);
        assert(y[0] == 0.0f && !std::signbit(y[0]));
        assert(y[1] == 0.0f &&  std::signbit(y[1]));
        assert(y[2] == 1.0f && y[3] == -1.0f);
        assert(std::isnan(y[4]));
        assert(y[5] == 1e-30f && y[6] == -1e-30f && y[7] == 1e-40f);
        assert(y[8] == 1.0f && y[9] == -1.0f);
    }

    check(act::sigmoid, "sigmoid", 80.0f, 4.0);
    {
        const std::vector<float> x = { 0.0f, INFINITY, -INFINITY, NAN, 100.0f };
        const std::vector<float> y = run(act::sigmoid, x, (int64_t) x.size(), 1);
        assert(y[0] == 0.5f);
        assert(y[1] == 1.0f && y[2] >= 0.0f && y[2] < 1e-37f);
        assert(std::isnan(y[3]));
        assert(y[4] == 1.0f);
    }

    printf("test-esn-activations: PASSED (A..C)\n");
    return 0;
}
