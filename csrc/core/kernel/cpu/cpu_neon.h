/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cpu_neon.h
 */

#ifdef ALLSPARK_USE_NEON_
#include <arm_neon.h>

#include <array>
#include <limits>

#define V_PARALLEL_THRESHOLD 1000

namespace allspark {
namespace cpu {
/** Exponent polynomial coefficients */
const std::array<float32x4_t, 8> exp_tab = {{
    vdupq_n_f32(1.f),
    vdupq_n_f32(0.0416598916054f),
    vdupq_n_f32(0.500000596046f),
    vdupq_n_f32(0.0014122662833f),
    vdupq_n_f32(1.00000011921f),
    vdupq_n_f32(0.00833693705499f),
    vdupq_n_f32(0.166665703058f),
    vdupq_n_f32(0.000195780929062f),
}};

/** Logarithm polynomial coefficients */
const std::array<float32x4_t, 8> log_tab = {{
    vdupq_n_f32(-2.29561495781f),
    vdupq_n_f32(-2.47071170807f),
    vdupq_n_f32(-5.68692588806f),
    vdupq_n_f32(-0.165253549814f),
    vdupq_n_f32(5.17591238022f),
    vdupq_n_f32(0.844007015228f),
    vdupq_n_f32(4.58445882797f),
    vdupq_n_f32(0.0141278216615f),
}};

/** Sin polynomial coefficients */
constexpr float te_sin_coeff2 = 0.166666666666f;  // 1/(2*3)
constexpr float te_sin_coeff3 = 0.05f;            // 1/(4*5)
constexpr float te_sin_coeff4 = 0.023809523810f;  // 1/(6*7)
constexpr float te_sin_coeff5 = 0.013888888889f;  // 1/(8*9)

inline float32x4_t vtaylor_polyq_f32(float32x4_t x,
                                     const std::array<float32x4_t, 8>& coeffs) {
  float32x4_t A = vmlaq_f32(coeffs[0], coeffs[4], x);
  float32x4_t B = vmlaq_f32(coeffs[2], coeffs[6], x);
  float32x4_t C = vmlaq_f32(coeffs[1], coeffs[5], x);
  float32x4_t D = vmlaq_f32(coeffs[3], coeffs[7], x);
  float32x4_t x2 = vmulq_f32(x, x);
  float32x4_t x4 = vmulq_f32(x2, x2);
  float32x4_t res = vmlaq_f32(vmlaq_f32(A, B, x2), vmlaq_f32(C, D, x2), x4);
  return res;
}

inline float32x4_t vinvsqrtq_f32(float32x4_t x) {
  float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);
  sqrt_reciprocal =
      vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal),
                sqrt_reciprocal);
  sqrt_reciprocal =
      vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal),
                sqrt_reciprocal);

  return sqrt_reciprocal;
}

inline float32x4_t vinvq_f32(float32x4_t x) {
  float32x4_t recip = vrecpeq_f32(x);
  recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
  recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
  return recip;
}

inline float32x4_t vsqrtq_f32(float32x4_t x) {
  return vmulq_f32(x, vinvsqrtq_f32(x));
}

inline float32x4_t vexpq_f32(float32x4_t x) {
  static const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f);  // ln(2)
  static const float32x4_t CONST_INV_LN2 =
      vdupq_n_f32(1.4426950408f);  // 1/ln(2)
  static const float32x4_t CONST_INF =
      vdupq_n_f32(std::numeric_limits<float>::infinity());
  static const float32x4_t CONST_MAX_INPUT = vdupq_n_f32(88.7f);
  static const float32x4_t CONST_0 = vdupq_n_f32(0.f);
  static const int32x4_t CONST_NEGATIVE_126 = vdupq_n_s32(-126);

  // Perform range reduction [-log(2),log(2)]
  int32x4_t m = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
  float32x4_t val = vmlsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);

  // Polynomial Approximation
  float32x4_t poly = vtaylor_polyq_f32(val, exp_tab);

  // Reconstruct
  poly = vreinterpretq_f32_s32(
      vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
  poly = vbslq_f32(vcltq_s32(m, CONST_NEGATIVE_126), CONST_0,
                   poly);  // Handle underflow
  poly = vbslq_f32(vcgtq_f32(x, CONST_MAX_INPUT), CONST_INF,
                   poly);  // Handle overflow

  return poly;
}

inline float32x4_t vlogq_f32(float32x4_t x) {
  static const int32x4_t CONST_127 = vdupq_n_s32(127);              // 127
  static const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f);  // ln(2)

  // Extract exponent
  int32x4_t m = vsubq_s32(
      vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23)),
      CONST_127);
  float32x4_t val = vreinterpretq_f32_s32(
      vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));

  // Polynomial Approximation
  float32x4_t poly = vtaylor_polyq_f32(val, log_tab);

  // Reconstruct
  poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);

  return poly;
}

inline float32x4_t vtanhq_f32(float32x4_t val) {
  static const float32x4_t CONST_1 = vdupq_n_f32(1.f);
  static const float32x4_t CONST_2 = vdupq_n_f32(2.f);
  static const float32x4_t CONST_MIN_TANH = vdupq_n_f32(-10.f);
  static const float32x4_t CONST_MAX_TANH = vdupq_n_f32(10.f);

  float32x4_t x = vminq_f32(vmaxq_f32(val, CONST_MIN_TANH), CONST_MAX_TANH);
  float32x4_t exp2x = vexpq_f32(vmulq_f32(CONST_2, x));
  float32x4_t num = vsubq_f32(exp2x, CONST_1);
  float32x4_t den = vaddq_f32(exp2x, CONST_1);
  float32x4_t tanh = vmulq_f32(num, vinvq_f32(den));
  return tanh;
}

inline float32x4_t vsigmoidq_f32(float32x4_t val) {
  static const float32x4_t CONST_1 = vdupq_n_f32(1.f);
  static const float32x4_t CONST_NEG_1 = vdupq_n_f32(-1.f);

  float32x4_t exp_x = vexpq_f32(vmulq_f32(CONST_NEG_1, val));
  float32x4_t den = vaddq_f32(exp_x, CONST_1);
  float32x4_t tanh = vinvq_f32(den);
  return tanh;
}

inline float32x4_t vsinq_f32(float32x4_t val) {
  const float32x4_t pi_v = vdupq_n_f32(M_PI);
  const float32x4_t pio2_v = vdupq_n_f32(M_PI / 2);
  const float32x4_t ipi_v = vdupq_n_f32(1 / M_PI);

  // Find positive or negative
  const int32x4_t c_v = vabsq_s32(vcvtq_s32_f32(vmulq_f32(val, ipi_v)));
  const uint32x4_t sign_v = vcleq_f32(val, vdupq_n_f32(0));
  const uint32x4_t odd_v =
      vandq_u32(vreinterpretq_u32_s32(c_v), vdupq_n_u32(1));

  uint32x4_t neg_v = veorq_u32(odd_v, sign_v);

  // Modulus a - (n * int(a*(1/n)))
  float32x4_t ma =
      vsubq_f32(vabsq_f32(val), vmulq_f32(pi_v, vcvtq_f32_s32(c_v)));
  const uint32x4_t reb_v = vcgeq_f32(ma, pio2_v);

  // Rebase a between 0 and pi/2
  ma = vbslq_f32(reb_v, vsubq_f32(pi_v, ma), ma);

  // Taylor series
  const float32x4_t ma2 = vmulq_f32(ma, ma);

  // 2nd elem: x^3 / 3!
  float32x4_t elem = vmulq_f32(vmulq_f32(ma, ma2), vdupq_n_f32(te_sin_coeff2));
  float32x4_t res = vsubq_f32(ma, elem);

  // 3rd elem: x^5 / 5!
  elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_sin_coeff3));
  res = vaddq_f32(res, elem);

  // 4th elem: x^7 / 7!float32x2_t vsin_f32(float32x2_t val)
  elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_sin_coeff4));
  res = vsubq_f32(res, elem);

  // 5th elem: x^9 / 9!
  elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_sin_coeff5));
  res = vaddq_f32(res, elem);

  // Change of sign
  neg_v = vshlq_n_u32(neg_v, 31);
  res = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(res), neg_v));
  return res;
}

}  // namespace cpu
}  // namespace allspark
#endif