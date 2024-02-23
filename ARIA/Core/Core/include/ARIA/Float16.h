#pragma once

#include "ARIA/ARIA.h"

#include <cuda/std/limits>
#include <cuda_fp16.h>

namespace ARIA {

using float16 = half;

//
//
//
//
//
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
  [[nodiscard]] static /*constexpr*/ ARIA::float16 min() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 max() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 lowest() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 epsilon() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 round_error() noexcept { return ARIA::float16{0.5F}; }

  [[nodiscard]] static /*constexpr*/ ARIA::float16 denorm_min() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 infinity() noexcept {
    return ARIA::float16{std::numeric_limits<float>::infinity()};
  }

  [[nodiscard]] static /*constexpr*/ ARIA::float16 quiet_NaN() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 signaling_NaN() noexcept = delete;

  static constexpr int digits = 10;
  // static constexpr int digits10 = ...;
  // static constexpr int max_digits10 = ...;
  static constexpr int max_exponent = 16;
  // static constexpr int max_exponent10 = ...;
  static constexpr int min_exponent = -13;
  // static constexpr int min_exponent10 = ...;
};

} // namespace std

//
//
//
namespace cuda::std {

template <>
class numeric_limits<ARIA::float16> : public ARIA::float16_::detail::aria_numeric_limits_float16_base {
public:
  [[nodiscard]] static /*constexpr*/ ARIA::float16 min() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 max() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 lowest() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 epsilon() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 round_error() noexcept { return ARIA::float16{0.5F}; }

  [[nodiscard]] static /*constexpr*/ ARIA::float16 denorm_min() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 infinity() noexcept {
    return ARIA::float16{std::numeric_limits<float>::infinity()};
  }

  [[nodiscard]] static /*constexpr*/ ARIA::float16 quiet_NaN() noexcept = delete;

  [[nodiscard]] static /*constexpr*/ ARIA::float16 signaling_NaN() noexcept = delete;

  static constexpr int digits = 10;
  // static constexpr int digits10 = ...;
  // static constexpr int max_digits10 = ...;
  static constexpr int max_exponent = 16;
  // static constexpr int max_exponent10 = ...;
  static constexpr int min_exponent = -13;
  // static constexpr int min_exponent10 = ...;
};

} // namespace cuda::std
