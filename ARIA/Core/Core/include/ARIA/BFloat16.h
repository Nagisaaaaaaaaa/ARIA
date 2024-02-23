#pragma once

/// \file
/// \brief A 16-bit half-precision floating point (FP16) implementation:
/// 1 sign bit, 5 exponent bits, and 10 mantissa bits.
/// See https://en.wikipedia.org/wiki/Half-precision_floating-point_format.
//
//
//
//
//
#include "ARIA/ARIA.h"

#include <cuda/std/limits>
#include <cuda_bf16.h>

namespace ARIA {

/// \brief A 16-bit half-precision floating point (FP16) implementation:
/// 1 sign bit, 5 exponent bits, and 10 mantissa bits.
///
/// ARIA directly uses CUDA's implementation,
/// see https://docs.nvidia.com/cuda/cuda-math-api/struct____half.html.
///
/// \example ```cpp
/// float16 a{0.1F};
/// float16 b{0.2F};
/// float16 c = a + b;
///
/// float16 inf0 = std::numeric_limits<float16>::infinity();
/// float16 inf1 = cuda::std::numeric_limits<float16>::infinity();
/// ```
using float16 = nv_bfloat16;

//
//
//
//
//
// Specialized numeric limits.
namespace float16_::detail {

struct aria_numeric_limits_float16_base_default {
  static constexpr std::float_denorm_style has_denorm = std::denorm_absent;
  static constexpr bool has_denorm_loss = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr bool is_bounded = false;
  static constexpr bool is_exact = false;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_integer = false;
  static constexpr bool is_modulo = false;
  static constexpr bool is_signed = false;
  static constexpr bool is_specialized = false;
  static constexpr bool tinyness_before = false;
  static constexpr bool traps = false;
  static constexpr std::float_round_style round_style = std::round_toward_zero;
  static constexpr int digits = 0;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 0;
  static constexpr int max_exponent = 0;
  static constexpr int max_exponent10 = 0;
  static constexpr int min_exponent = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int radix = 0;
};

struct aria_numeric_limits_float16_base : aria_numeric_limits_float16_base_default {
  static constexpr std::float_denorm_style has_denorm = std::denorm_present;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr std::float_round_style round_style = std::round_to_nearest;
  static constexpr int radix = 2;
};

} // namespace float16_::detail

} // namespace ARIA

//
//
//
namespace std {

template <>
class numeric_limits<ARIA::float16> : public ARIA::float16_::detail::aria_numeric_limits_float16_base {
public:
  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 min() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 max() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 lowest() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 epsilon() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 round_error() noexcept {
    return ARIA::float16{0.5F};
  }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 denorm_min() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 infinity() noexcept {
    return ARIA::float16{std::numeric_limits<float>::infinity()};
  }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 quiet_NaN() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 signaling_NaN() noexcept = delete;

  static constexpr int digits = 10;
  // static constexpr int digits10 = ...;
  // static constexpr int max_digits10 = ...;
  static constexpr int max_exponent = 16;
  // static constexpr int max_exponent10 = ...;
  static constexpr int min_exponent = -13;
  // static constexpr int min_exponent10 = ...;
};

} // namespace std

namespace cuda::std {

template <>
class numeric_limits<ARIA::float16> : public ::std::numeric_limits<ARIA::float16> {};

} // namespace cuda::std
