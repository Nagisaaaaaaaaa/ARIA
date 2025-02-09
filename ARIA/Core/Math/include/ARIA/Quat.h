#pragma once

/// \file
/// \brief ARIA directly uses `Eigen` as linear algebra library.
/// Make sure you are familiar with `Eigen` before continue.
/// See https://eigen.tuxfamily.org.
///
/// A `Quat` is a direct using of the `Eigen::Quaternion`.

//
//
//
//
//
#include "ARIA/Math.h"
#include "ARIA/Vec.h"
#include "ARIA/detail/QuatImpl.h"

namespace ARIA {

/// \brief A direct using of the `Eigen::Quaternion`.
template <typename T>
using Quat = quat::detail::Quat<T>;

//
//
//
using Quatf = Quat<float>;
using Quatd = Quat<double>;
using Quatr = Quat<Real>;

//
//
//
//
//
/// \brief Convert from quaternion to Euler angles.
///
/// \example ```cpp
/// Quatr q = Quatr::Identity();
/// Vec3r euler = ToEulerAngles(q);
/// ```
///
/// \details This implementation is based on glm, see https://github.com/g-truc/glm.
template <typename TInput>
[[nodiscard]] static inline constexpr auto ToEulerAngles(const TInput &qIn) noexcept;

/// \brief Convert from Euler angles to quaternion.
///
/// \example ```cpp
/// Quatr q1 = FromEulerAngles(Vec3r(10_R, 20_R, 30_R) * deg2Rad<Real>);
/// ```
template <typename TInput>
[[nodiscard]] static inline constexpr auto FromEulerAngles(const TInput &eulerIn) noexcept;

//
//
//
//
//
/// \brief A property prefab for type `class Quat`.
/// All possible sub-properties and functions have been defined here.
///
/// \example ```cpp
/// ARIA_PROP_PREFAB_QUAT(public, public, __host__, Quatr, rotation)
/// ```
#define ARIA_PROP_PREFAB_QUAT /*(accessGet, accessSet, specifiers, type, propName, (optional) args...)*/               \
  __ARIA_PROP_PREFAB_QUAT

/// \brief A sub-property prefab for type `class Quat`.
/// All possible sub-sub-properties and functions have been defined here.
///
/// \example ```cpp
/// ARIA_SUB_PROP_PREFAB_QUAT(__host__, Quatr, rotation)
/// ```
#define ARIA_SUB_PROP_PREFAB_QUAT(specifiers, type, propName) __ARIA_SUB_PROP_PREFAB_QUAT(specifiers, type, propName)

//
//
//
//
//
/// \brief The built-in `Mosaic` for `Quat`.
/// See `Mosaic.h`, `Array.h`, and `Vector.h` for more details.
///
/// \example ```cpp
/// using TMosaic0 = QuatMosaic<Real>;
/// using TMosaic1 = QuatMosaic<Quatr>; // The same type as `TMosaic0`.
/// using TMosaic2 = mosaic_t<Quatr>;   // Also the same.
/// ```
template <typename T>
using QuatMosaic = quat::detail::reduce_quat_mosaic_t<T>;

template <typename T>
struct mosaic::detail::MosaicBuiltIn<Quat<T>> {
  using type = QuatMosaic<T>;
};

//
//
//
//
//
//
//
//
//
template <typename TInput>
[[nodiscard]] static inline constexpr auto ToEulerAngles(const TInput &qIn) noexcept {
  //! The input parameter is not `const Quat<T>& q` in order to support `Eigen` proxies.
  //! That is also why the input parameter is tried and converted to `Quat` with `Auto`.
  auto q = Auto(qIn);
  static_assert(quat::detail::is_quat_v<decltype(q)>, "The input should be convertible to `Quat`");
  using T = decltype(q)::Scalar;

  Vec3<T> angles;

  // Roll (x-axis rotation).
  const T sinRCosP = 2 * (q.w() * q.x() + q.y() * q.z());
  const T cosRSinP = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
  angles.x() = std::atan2(sinRCosP, cosRSinP);

  // Pitch (y-axis rotation).
  const T sinP = 2 * (q.w() * q.y() - q.z() * q.x());
  if (std::abs(sinP) >= 1)
    angles.y() = std::copysign(pi<T> / 2, sinP); // Use 90 degrees if out of range.
  else
    angles.y() = std::asin(sinP);

  // Yaw (z-axis rotation).
  const T sinYCosP = 2 * (q.w() * q.z() + q.x() * q.y());
  const T cosYCosP = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
  angles.z() = std::atan2(sinYCosP, cosYCosP);

  return angles;
}

template <typename TInput>
[[nodiscard]] static inline constexpr auto FromEulerAngles(const TInput &eulerIn) noexcept {
  //! The input parameter is not `const Vec3<T>& q` in order to support `Eigen` proxies.
  //! That is also why the input parameter is tried and converted to `Vec` with `Auto`.
  auto euler = Auto(eulerIn);
  static_assert(vec::detail::is_vec_s_v<decltype(euler), 3>, "The input should be convertible to `Vec3`");
  using T = decltype(euler)::Scalar;

  using Eigen::AngleAxis;

  const AngleAxis<T> rollAngle(euler.x(), Vec3<T>::UnitX());
  const AngleAxis<T> pitchAngle(euler.y(), Vec3<T>::UnitY());
  const AngleAxis<T> yawAngle(euler.z(), Vec3<T>::UnitZ());
  const Quat<T> quat = yawAngle * pitchAngle * rollAngle;

  return quat;
}

} // namespace ARIA
