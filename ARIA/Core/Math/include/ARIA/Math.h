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
