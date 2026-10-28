// SPDX-License-Identifier: MIT
//
// Faster CPU dot product for the BPDQ2 2-bit weight format
// (diegogranziol/whisper-large-v3-turbo-bpdq-2bit-qat).
//
// This file is #included into ggml/src/ggml-cpu/ggml-cpu.c by
// prepare_whisper.cmake, right before the patch's scalar
// ggml_vec_dot_bpdq2_f32(). It expects these to be in scope:
//   block_bpdq2, QK_BPDQ2, GGML_RESTRICT, GGML_ASSERT, GGML_UNUSED,
//   GGML_FP16_TO_FP32
//
// Block layout (72 bytes, 256 weights):
//   ggml_half levels[4];   // four FP16 reconstruction values
//   uint32_t  codes[16];   // weight j -> (codes[j/16] >> 2*(j%16)) & 3
//
// nrc == 1: *s = dot(x, y), exactly like the patch's scalar kernel.
// nrc == 2: 2x2 tile, used by ggml's mul_mat when the BPDQ2 type is registered
//           with .nrows = 2 (the same convention as ggml's ARM i8mm kernels):
//             s[0]  = x0.y0   s[1]    = x1.y0
//             s[bs] = x0.y1   s[bs+1] = x1.y1
//           where x1 = x0 + bx bytes and y1 = y0 + by bytes. Each weight block
//           is decoded once and used for two activation columns.
//
// The sums are the same as the reference loop, only accumulated in a
// different order (several FP32 lanes/chains).

// BPDQ2_FORCE_NEON is only used by the host unit test (NEON emulated via SIMDe).
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(BPDQ2_FORCE_NEON)
#include <arm_neon.h>

typedef struct {
    int32x4_t  sh[4];   // per-lane right shifts for the 4 groups of 4 codes in a word
    uint32x4_t three;
    uint32x4_t bytes4;  // code c -> byte offset 4c, replicated in every byte
    uint32x4_t lane;    // + bytes 0..3 of the selected float
} bpdq2_consts;

static inline bpdq2_consts bpdq2_make_consts(void) {
    static const int32_t shifts[16] = {
          0,  -2,  -4,  -6,
         -8, -10, -12, -14,
        -16, -18, -20, -22,
        -24, -26, -28, -30,
    };
    bpdq2_consts k;
    k.sh[0]  = vld1q_s32(shifts +  0);
    k.sh[1]  = vld1q_s32(shifts +  4);
    k.sh[2]  = vld1q_s32(shifts +  8);
    k.sh[3]  = vld1q_s32(shifts + 12);
    k.three  = vdupq_n_u32(3);
    k.bytes4 = vdupq_n_u32(0x04040404u);
    k.lane   = vdupq_n_u32(0x03020100u);
    return k;
}

// 16-byte lookup table holding a block's four FP32 levels.
static inline uint8x16_t bpdq2_lut(const block_bpdq2 * GGML_RESTRICT b) {
    const float f[4] = {
        GGML_FP16_TO_FP32(b->levels[0]),
        GGML_FP16_TO_FP32(b->levels[1]),
        GGML_FP16_TO_FP32(b->levels[2]),
        GGML_FP16_TO_FP32(b->levels[3]),
    };
    return vreinterpretq_u8_f32(vld1q_f32(f));
}

// Decode 4 consecutive weights (group g of a 16-code word) to FP32.
static inline float32x4_t bpdq2_decode4(const bpdq2_consts * k, uint8x16_t lut, uint32x4_t cw, int g) {
    const uint32x4_t c = vandq_u32(vshlq_u32(cw, k->sh[g]), k->three);
    return vreinterpretq_f32_u8(vqtbl1q_u8(lut, vreinterpretq_u8_u32(vmlaq_u32(k->lane, c, k->bytes4))));
}

static void ggml_vec_dot_bpdq2_f32_fast(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_ASSERT(n % QK_BPDQ2 == 0 && (nrc == 1 || nrc == 2));
    const int nb = n / QK_BPDQ2;
    const bpdq2_consts k = bpdq2_make_consts();

    if (nrc == 1) {
        const block_bpdq2 * GGML_RESTRICT x = (const block_bpdq2 *) vx;
        const float       * GGML_RESTRICT y = (const float *) vy;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        for (int b = 0; b < nb; ++b) {
            const uint8x16_t lut = bpdq2_lut(&x[b]);
            const float * GGML_RESTRICT yb = y + (size_t) b * QK_BPDQ2;
            for (int w = 0; w < 16; ++w) {
                const uint32x4_t cw = vdupq_n_u32(x[b].codes[w]);
                const float * GGML_RESTRICT yw = yb + 16*w;
                acc0 = vfmaq_f32(acc0, bpdq2_decode4(&k, lut, cw, 0), vld1q_f32(yw +  0));
                acc1 = vfmaq_f32(acc1, bpdq2_decode4(&k, lut, cw, 1), vld1q_f32(yw +  4));
                acc2 = vfmaq_f32(acc2, bpdq2_decode4(&k, lut, cw, 2), vld1q_f32(yw +  8));
                acc3 = vfmaq_f32(acc3, bpdq2_decode4(&k, lut, cw, 3), vld1q_f32(yw + 12));
            }
        }
        *s = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    } else {
        const block_bpdq2 * GGML_RESTRICT x0 = (const block_bpdq2 *) vx;
        const block_bpdq2 * GGML_RESTRICT x1 = (const block_bpdq2 *) ((const char *) vx + bx);
        const float       * GGML_RESTRICT y0 = (const float *) vy;
        const float       * GGML_RESTRICT y1 = (const float *) ((const char *) vy + by);

        // a_rc: weight row r against activation column c; two chains each (even/odd groups)
        float32x4_t a00e = vdupq_n_f32(0.0f), a00o = vdupq_n_f32(0.0f);
        float32x4_t a10e = vdupq_n_f32(0.0f), a10o = vdupq_n_f32(0.0f);
        float32x4_t a01e = vdupq_n_f32(0.0f), a01o = vdupq_n_f32(0.0f);
        float32x4_t a11e = vdupq_n_f32(0.0f), a11o = vdupq_n_f32(0.0f);

        for (int b = 0; b < nb; ++b) {
            const uint8x16_t lut0 = bpdq2_lut(&x0[b]);
            const uint8x16_t lut1 = bpdq2_lut(&x1[b]);
            const float * GGML_RESTRICT y0b = y0 + (size_t) b * QK_BPDQ2;
            const float * GGML_RESTRICT y1b = y1 + (size_t) b * QK_BPDQ2;
            for (int w = 0; w < 16; ++w) {
                const uint32x4_t c0 = vdupq_n_u32(x0[b].codes[w]);
                const uint32x4_t c1 = vdupq_n_u32(x1[b].codes[w]);
                const float * GGML_RESTRICT p0 = y0b + 16*w;
                const float * GGML_RESTRICT p1 = y1b + 16*w;
                for (int g = 0; g < 4; g += 2) {
                    const float32x4_t w0e = bpdq2_decode4(&k, lut0, c0, g);
                    const float32x4_t w1e = bpdq2_decode4(&k, lut1, c1, g);
                    const float32x4_t w0o = bpdq2_decode4(&k, lut0, c0, g + 1);
                    const float32x4_t w1o = bpdq2_decode4(&k, lut1, c1, g + 1);
                    const float32x4_t y0e = vld1q_f32(p0 + 4*g), y0o = vld1q_f32(p0 + 4*g + 4);
                    const float32x4_t y1e = vld1q_f32(p1 + 4*g), y1o = vld1q_f32(p1 + 4*g + 4);
                    a00e = vfmaq_f32(a00e, w0e, y0e);  a00o = vfmaq_f32(a00o, w0o, y0o);
                    a10e = vfmaq_f32(a10e, w1e, y0e);  a10o = vfmaq_f32(a10o, w1o, y0o);
                    a01e = vfmaq_f32(a01e, w0e, y1e);  a01o = vfmaq_f32(a01o, w0o, y1o);
                    a11e = vfmaq_f32(a11e, w1e, y1e);  a11o = vfmaq_f32(a11o, w1o, y1o);
                }
            }
        }
        s[0]      = vaddvq_f32(vaddq_f32(a00e, a00o));
        s[1]      = vaddvq_f32(vaddq_f32(a10e, a10o));
        s[bs]     = vaddvq_f32(vaddq_f32(a01e, a01o));
        s[bs + 1] = vaddvq_f32(vaddq_f32(a11e, a11o));
    }
}

#else // no AArch64 NEON: same arithmetic as the reference, kept so the build never breaks

static float bpdq2_dot_ref(int n, const block_bpdq2 * x, const float * y) {
    float sum = 0.0f;
    for (int b = 0; b < n / QK_BPDQ2; ++b) {
        float lut[4];
        for (int c = 0; c < 4; ++c) lut[c] = GGML_FP16_TO_FP32(x[b].levels[c]);
        for (int j = 0; j < QK_BPDQ2; ++j) {
            sum += lut[(x[b].codes[j/16] >> (2*(j%16))) & 3] * y[b*QK_BPDQ2 + j];
        }
    }
    return sum;
}

static void ggml_vec_dot_bpdq2_f32_fast(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_ASSERT(n % QK_BPDQ2 == 0 && (nrc == 1 || nrc == 2));
    for (int r = 0; r < nrc; ++r) {          // activation column
        for (int c = 0; c < nrc; ++c) {      // weight row
            const block_bpdq2 * x = (const block_bpdq2 *) ((const char *) vx + c*bx);
            const float * y = (const float *) ((const char *) vy + r*by);
            s[r*bs + c] = bpdq2_dot_ref(n, x, y);
        }
    }
}

#endif
