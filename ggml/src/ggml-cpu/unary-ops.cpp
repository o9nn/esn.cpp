#include "unary-ops.h"

#include <cstdint>
#include <cstring>

static inline float op_abs(float x) {
    return fabsf(x);
}

static inline float op_sgn(float x) {
    return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

static inline float op_neg(float x) {
    return -x;
}

static inline float op_step(float x) {
    return (x > 0.f) ? 1.f : 0.f;
}

// expf without a libm call, so that the loops in tanh and sigmoid below
// auto-vectorize: x = n*ln2 + r with |r| <= ln2/2, exp(r) by the Cephes
// polynomial and n added to the exponent bits. Within 1 ulp of exp on the
// clamped range [-86.6, 88.3], which keeps 2^n a normal float; NaN maps to
// the lower bound (callers that care propagate it themselves).
static inline float op_expf_poly(float x) {
    x = x > -86.6f ? x : -86.6f;
    x = x <  88.3f ? x :  88.3f;
    const float n = (x * 1.44269504088896341f + 12582912.0f) - 12582912.0f; // round to nearest
    const float r = (x - n * 0.693359375f) + n * 2.12194440e-4f;              // x - n*ln2 in two parts
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = (p * r) * r + r + 1.0f;
    int32_t bits;
    memcpy(&bits, &p, sizeof(bits));
    bits += (int32_t) n << 23;
    memcpy(&p, &bits, sizeof(p));
    return p;
}

// Branch-free tanh, within 1 ulp like libm's tanhf but vectorizable (scalar
// tanhf dominates small recurrent graphs, e.g. echo state networks).
// |x| < 0.625: x*P(x^2), a minimax fit of tanh(x)/x; above it
// 1 - 2/(exp(2|x|) + 1), whose cancellation is harmless there. |x| is capped
// at 10, past where tanh rounds to 1.
static inline float op_tanh(float x) {
    const float ax = fabsf(x);
    const float a  = ax < 10.0f ? ax : 10.0f;
    const float a2 = a * a;
    float q = 2.1286243572831154e-3f;
    q = q * a2 - 8.15926119685173e-3f;
    q = q * a2 + 2.169189229607582e-2f;
    q = q * a2 - 5.394471436738968e-2f;
    q = q * a2 + 1.3333183526992798e-1f;
    q = q * a2 - 3.33333283662796e-1f;
    const float small = (q * a2) * a + a;
    const float large = 1.0f - 2.0f / (op_expf_poly(2.0f * a) + 1.0f);
    const float t = copysignf(a < 0.625f ? small : large, x);
    return x != x ? x : t;
}

static inline float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static inline float op_relu(float x) {
    return (x > 0.f) ? x : 0.f;
}

static inline float op_sigmoid(float x) {
    const float s = 1.f / (1.f + op_expf_poly(-x));
    return x != x ? x : s;
}

static inline float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_exp(float x) {
    return expf(x);
}

static inline float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_sqr(float x) {
    return x * x;
}

static inline float op_sqrt(float x) {
    return sqrtf(x);
}

static inline float op_xielu(float x, float alpha_n, float alpha_p, float beta, float eps) {
    if (x > 0.0f) {
        return alpha_p * x * x + beta * x;
    } else {
        const float min_x_eps = fminf(x, eps);
        return (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

static inline float op_sin(float x) {
    return sinf(x);
}

static inline float op_cos(float x) {
    return cosf(x);
}

static inline float op_log(float x) {
    return logf(x);
}

template <float (*op)(float), typename src0_t, typename dst_t>
static inline void vec_unary_op(int64_t n, dst_t * y, const src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <float (*op)(float), typename src0_t, typename dst_t>
static void apply_unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float)>
static void unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <float (*op)(float, ggml_tensor *)>
static void unary_op_params(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static inline void vec_unary_op_functor(int64_t n, dst_t * y, const src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static void apply_unary_op_functor(const ggml_compute_params * params, ggml_tensor * dst, Op op) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
static void unary_op_functor(const ggml_compute_params * params, ggml_tensor * dst, Op op) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, float, float>(params, dst, op);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, float>(params, dst, op);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, float>(params, dst, op);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_compute_forward_abs(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

void ggml_compute_forward_sgn(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

void ggml_compute_forward_neg(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

void ggml_compute_forward_step(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

void ggml_compute_forward_tanh(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

void ggml_compute_forward_elu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

void ggml_compute_forward_relu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

void ggml_compute_forward_sigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

void ggml_compute_forward_hardsigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

void ggml_compute_forward_exp(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

void ggml_compute_forward_hardswish(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

void ggml_compute_forward_sqr(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

void ggml_compute_forward_sqrt(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

void ggml_compute_forward_sin(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

void ggml_compute_forward_cos(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

void ggml_compute_forward_log(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

void ggml_compute_forward_xielu(const ggml_compute_params * params, ggml_tensor * dst) {
    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta = ggml_get_op_params_f32(dst, 3);
    const float eps = ggml_get_op_params_f32(dst, 4);

    const auto xielu_op_params = [alpha_n, alpha_p, beta, eps](float f) {
        return op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

