#pragma once

namespace ARIA {

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

// TODO: Implement all the functions below (even though they are seldom used).
template <>
class numeric_limits<ARIA::bfloat16> : public ARIA::bfloat16_::detail::numeric_limits_bfloat16_base {
public:
  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 min() noexcept { return CUDART_MIN_DENORM_BF16; }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 max() noexcept { return CUDART_MAX_NORMAL_BF16; }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 lowest() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 epsilon() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 round_error() noexcept {
    return ARIA::bfloat16{0.5F};
  }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 denorm_min() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 infinity() noexcept { return CUDART_INF_BF16; }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::bfloat16 quiet_NaN() noexcept { return CUDART_NAN_BF16; }

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

//
//
//
//
//
ARIA_HOST_DEVICE ARIA::bfloat16 abs(const ARIA::bfloat16 &x) {
  return __habs(x);
}

ARIA_HOST_DEVICE ARIA::bfloat16 max(const ARIA::bfloat16 &x, const ARIA::bfloat16 &y) {
  return __hmax(x, y);
}

ARIA_HOST_DEVICE ARIA::bfloat16 min(const ARIA::bfloat16 &x, const ARIA::bfloat16 &y) {
  return __hmin(x, y);
}

namespace std {

ARIA_HOST_DEVICE ARIA::bfloat16 abs(const ARIA::bfloat16 &x) {
  return ::abs(x);
}

ARIA_HOST_DEVICE ARIA::bfloat16 max(const ARIA::bfloat16 &x, const ARIA::bfloat16 &y) {
  return ::max(x, y);
}

ARIA_HOST_DEVICE ARIA::bfloat16 min(const ARIA::bfloat16 &x, const ARIA::bfloat16 &y) {
  return ::min(x, y);
}

} // namespace std

namespace cuda::std {

ARIA_HOST_DEVICE ARIA::bfloat16 abs(const ARIA::bfloat16 &x) {
  return ::abs(x);
}

ARIA_HOST_DEVICE ARIA::bfloat16 max(const ARIA::bfloat16 &x, const ARIA::bfloat16 &y) {
  return ::max(x, y);
}

ARIA_HOST_DEVICE ARIA::bfloat16 min(const ARIA::bfloat16 &x, const ARIA::bfloat16 &y) {
  return ::min(x, y);
}

} // namespace cuda::std
