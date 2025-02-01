#pragma once

namespace ARIA {

// Specialized numeric limits.
namespace float16_::detail {

struct numeric_limits_float16_base_default {
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

struct numeric_limits_float16_base : numeric_limits_float16_base_default {
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

// TODO: Implement all the functions below (even though they are seldom used).
template <>
class numeric_limits<ARIA::float16> : public ARIA::float16_::detail::numeric_limits_float16_base {
public:
  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 min() noexcept { return CUDART_MIN_DENORM_FP16; }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 max() noexcept { return CUDART_MAX_NORMAL_FP16; }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 lowest() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 epsilon() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 round_error() noexcept {
    return ARIA::float16{0.5F};
  }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 denorm_min() noexcept = delete;

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 infinity() noexcept { return CUDART_INF_FP16; }

  [[nodiscard]] ARIA_HOST_DEVICE static /*constexpr*/ ARIA::float16 quiet_NaN() noexcept { return CUDART_NAN_FP16; }

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

//
//
//
//
//
[[nodiscard]] ARIA_HOST_DEVICE inline ARIA::float16 abs(const ARIA::float16 &x) {
  return __habs(x);
}

[[nodiscard]] ARIA_HOST_DEVICE inline ARIA::float16 max(const ARIA::float16 &x, const ARIA::float16 &y) {
  return __hmax(x, y);
}

[[nodiscard]] ARIA_HOST_DEVICE inline ARIA::float16 min(const ARIA::float16 &x, const ARIA::float16 &y) {
  return __hmin(x, y);
}

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
[[nodiscard]] ARIA_DEVICE inline ARIA::float16 ceil(const ARIA::float16 &x) {
  return hceil(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 cos(const ARIA::float16 &x) {
  return hcos(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 exp(const ARIA::float16 &x) {
  return hexp(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 exp10(const ARIA::float16 &x) {
  return hexp10(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 exp2(const ARIA::float16 &x) {
  return hexp2(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 floor(const ARIA::float16 &x) {
  return hfloor(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 log(const ARIA::float16 &x) {
  return hlog(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 log10(const ARIA::float16 &x) {
  return hlog10(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 log2(const ARIA::float16 &x) {
  return hlog2(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 rcp(const ARIA::float16 &x) {
  return hrcp(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 rint(const ARIA::float16 &x) {
  return hrint(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 rsqrt(const ARIA::float16 &x) {
  return hrsqrt(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 sin(const ARIA::float16 &x) {
  return hsin(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 sqrt(const ARIA::float16 &x) {
  return hsqrt(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 trunc(const ARIA::float16 &x) {
  return htrunc(x);
}
#endif

//
//
//
//! STL is extended, even though we know that it may be dangerous.
namespace std {

[[nodiscard]] ARIA_HOST_DEVICE inline ARIA::float16 abs(const ARIA::float16 &x) {
  return ::abs(x);
}

[[nodiscard]] ARIA_HOST_DEVICE inline ARIA::float16 max(const ARIA::float16 &x, const ARIA::float16 &y) {
  return ::max(x, y);
}

[[nodiscard]] ARIA_HOST_DEVICE inline ARIA::float16 min(const ARIA::float16 &x, const ARIA::float16 &y) {
  return ::min(x, y);
}

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
[[nodiscard]] ARIA_DEVICE inline ARIA::float16 ceil(const ARIA::float16 &x) {
  return ::ceil(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 cos(const ARIA::float16 &x) {
  return ::cos(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 exp(const ARIA::float16 &x) {
  return ::exp(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 exp10(const ARIA::float16 &x) {
  return ::exp10(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 exp2(const ARIA::float16 &x) {
  return ::exp2(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 floor(const ARIA::float16 &x) {
  return ::floor(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 log(const ARIA::float16 &x) {
  return ::log(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 log10(const ARIA::float16 &x) {
  return ::log10(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 log2(const ARIA::float16 &x) {
  return ::log2(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 rcp(const ARIA::float16 &x) {
  return ::rcp(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 rint(const ARIA::float16 &x) {
  return ::rint(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 rsqrt(const ARIA::float16 &x) {
  return ::rsqrt(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 sin(const ARIA::float16 &x) {
  return ::sin(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 sqrt(const ARIA::float16 &x) {
  return ::sqrt(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 trunc(const ARIA::float16 &x) {
  return ::trunc(x);
}
#endif

} // namespace std

//
//
//
#if 0 // TODO: No longer needed by newer version of CUDA.
namespace cuda::std {

[[nodiscard]] ARIA_HOST_DEVICE inline ARIA::float16 abs(const ARIA::float16 &x) {
  return ::abs(x);
}

[[nodiscard]] ARIA_HOST_DEVICE inline ARIA::float16 max(const ARIA::float16 &x, const ARIA::float16 &y) {
  return ::max(x, y);
}

[[nodiscard]] ARIA_HOST_DEVICE inline ARIA::float16 min(const ARIA::float16 &x, const ARIA::float16 &y) {
  return ::min(x, y);
}

  #if defined(__CUDACC__) || defined(_NVHPC_CUDA)
[[nodiscard]] ARIA_DEVICE inline ARIA::float16 ceil(const ARIA::float16 &x) {
  return ::ceil(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 cos(const ARIA::float16 &x) {
  return ::cos(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 exp(const ARIA::float16 &x) {
  return ::exp(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 exp10(const ARIA::float16 &x) {
  return ::exp10(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 exp2(const ARIA::float16 &x) {
  return ::exp2(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 floor(const ARIA::float16 &x) {
  return ::floor(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 log(const ARIA::float16 &x) {
  return ::log(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 log10(const ARIA::float16 &x) {
  return ::log10(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 log2(const ARIA::float16 &x) {
  return ::log2(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 rcp(const ARIA::float16 &x) {
  return ::rcp(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 rint(const ARIA::float16 &x) {
  return ::rint(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 rsqrt(const ARIA::float16 &x) {
  return ::rsqrt(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 sin(const ARIA::float16 &x) {
  return ::sin(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 sqrt(const ARIA::float16 &x) {
  return ::sqrt(x);
}

[[nodiscard]] ARIA_DEVICE inline ARIA::float16 trunc(const ARIA::float16 &x) {
  return ::trunc(x);
}
  #endif

} // namespace cuda::std
#endif

//
//
//
//
//
// Python.
namespace ARIA {

ARIA_PYTHON_TYPE_BEGIN(float16);
// Constructors.
ARIA_PYTHON_TYPE_CONSTRUCTOR(int8);
ARIA_PYTHON_TYPE_CONSTRUCTOR(uint8);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int16);
ARIA_PYTHON_TYPE_CONSTRUCTOR(uint16);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int);
ARIA_PYTHON_TYPE_CONSTRUCTOR(uint);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int64);
ARIA_PYTHON_TYPE_CONSTRUCTOR(uint64);
ARIA_PYTHON_TYPE_CONSTRUCTOR(float);
ARIA_PYTHON_TYPE_CONSTRUCTOR(double);
// Operators.
ARIA_PYTHON_TYPE_BINARY_OPERATOR(==);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(!=);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(<);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(>);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(<=);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(>=);
ARIA_PYTHON_TYPE_UNARY_OPERATOR(+);
ARIA_PYTHON_TYPE_UNARY_OPERATOR(-);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(+);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(-);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(*);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(/);
// External functions.
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(abs, const T &);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(max, const T &, const T &);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(min, const T &, const T &);
//
ARIA_PYTHON_TYPE_END;

} // namespace ARIA
