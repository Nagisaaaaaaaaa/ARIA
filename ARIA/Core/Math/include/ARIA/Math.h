#pragma once

/// \file
/// \brief This file includes the most basic mathematics-related constants and functions.

//
//
//
//
//
#include "ARIA/ARIA.h"

#include <numbers>

namespace ARIA {

/// \brief Get the positive infinity of the given universal number type.
template <typename T>
  requires(!std::integral<T>) //! Should not be `std::floating_point` here to support tailored universal number types.
static constexpr T infinity = std::numeric_limits<T>::infinity();

/// \brief Get the maximum value of the given type.
///
/// \warning For universal number types such as `float` and `double`,
/// `maximum` will return the largest finite value, not `infinity`.
template <typename T>
static constexpr T maximum = std::numeric_limits<T>::max();

/// \brief Get the minimum value of the given type.
///
/// \warning For universal number types such as `float` and `double`,
/// `minimum` will return the smallest finite value, not `-infinity`.
template <typename T>
static constexpr T minimum = std::numeric_limits<T>::min();

/// \brief `supremum` is mathematically defined as https://en.wikipedia.org/wiki/Infimum_and_supremum.
/// For `integral` types, returns `maximum`; for universal number types, returns `infinity`.
template <typename T>
static constexpr T supremum = std::integral<T> ? maximum<T> : std::numeric_limits<T>::infinity();

/// \brief `infimum` is mathematically defined as https://en.wikipedia.org/wiki/Infimum_and_supremum.
/// For `integral` types, returns `minimum`; for universal number types, returns `-infinity`.
template <typename T>
static constexpr T infimum = std::integral<T> ? minimum<T> : -std::numeric_limits<T>::infinity();

//
//
//
//
//
/// \brief 3.1415926...
template <typename T>
static constexpr T pi = std::numbers::pi_v<T>;

/// \brief 1 / 3.1415926...
template <typename T>
static constexpr T piInv = std::numbers::inv_pi_v<T>;

/// \brief 2.7182818284...
template <typename T>
static constexpr T e = std::numbers::e_v<T>;

//
//
//
//
//
/// \brief Degrees to radians.
///
/// \example ```cpp
/// Real rad = 180_R * deg2Rad<Real>;
/// Vec3r rad = Vec3r{180_R, 90_R, 45_R} * deg2Rad<Real>;
/// ```
template <typename T>
static constexpr T deg2Rad = static_cast<T>(pi<double> / 180.0);

/// \brief Radians to degrees.
///
/// \example ```cpp
/// Real deg = pi<Real> * rad2Deg<Real>;
/// Vec3r deg = Vec3r{pi<Real>, pi<Real> / 2, pi<Real> / 4} * rad2Deg<Real>;
/// ```
template <typename T>
static constexpr T rad2Deg = static_cast<T>(180.0 / pi<double>);

//
//
//
//
//
/// \brief Linear interpolation is defined as `x + t * (y - x)`.
///
/// \example ```cpp
template <typename Tx, typename Ty, typename Tt>
ARIA_HOST_DEVICE static inline constexpr auto Lerp(const Tx &x, const Ty &y, const Tt &t) {
  return x + t * (y - x);
}

} // namespace ARIA
