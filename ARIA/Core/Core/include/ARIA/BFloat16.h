#pragma once

/// \file
/// \brief A 16-bit brain floating point (BF16) implementation:
/// 1 sign bit, 8 exponent bits, and 7 mantissa bits.
/// See https://en.wikipedia.org/wiki/Bfloat16_floating-point_format.
//
//
//
//
//
#include "ARIA/ARIA.h"

#include <cuda/std/limits>
#include <cuda_bf16.h>

namespace ARIA {

/// \brief A 16-bit brain floating point (BF16) implementation:
/// 1 sign bit, 8 exponent bits, and 7 mantissa bits.
///
/// ARIA directly uses CUDA's implementation,
/// see https://docs.nvidia.com/cuda/cuda-math-api/struct____nv__bfloat16.html.
///
/// \example ```cpp
/// bfloat16 a{0.1F};
/// bfloat16 b{0.2F};
/// bfloat16 c = a + b;
///
/// bfloat16 inf0 = std::numeric_limits<bfloat16>::infinity();
/// bfloat16 inf1 = cuda::std::numeric_limits<bfloat16>::infinity();
/// ```
using bfloat16 = nv_bfloat16;

//
//
//
//
//
// Specialized numeric limits.
namespace bfloat16_::detail {

struct numeric_limits_bfloat16_base_default {
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

struct numeric_limits_bfloat16_base : numeric_limits_bfloat16_base_default {
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

} // namespace bfloat16_::detail

} // namespace ARIA

//
//
//
namespace std {

template <>
class numeric_limits<ARIA::bfloat16> : public ARIA::bfloat16_::detail::numeric_limits_bfloat16_base {
public:
  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 min() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 max() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 lowest() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 epsilon() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 round_error() noexcept {
    return ARIA::bfloat16{0.5F};
  }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 denorm_min() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 infinity() noexcept {
    return ARIA::bfloat16{std::numeric_limits<float>::infinity()};
  }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 quiet_NaN() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 signaling_NaN() noexcept = delete;

  static constexpr int digits = 7;
  // static constexpr int digits10 = ...;
  // static constexpr int max_digits10 = ...;
  static constexpr int max_exponent = 128;
  // static constexpr int max_exponent10 = ...;
  static constexpr int min_exponent = -125;
  // static constexpr int min_exponent10 = ...;
};

} // namespace std

namespace cuda::std {

template <>
class numeric_limits<ARIA::bfloat16> : public ::std::numeric_limits<ARIA::bfloat16> {};

} // namespace cuda::std
