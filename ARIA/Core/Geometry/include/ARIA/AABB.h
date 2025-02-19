#pragma once

/// \file
/// \brief An axis-aligned bounding box (AABB) implementation.
///
/// `AABB` is implemented generically in dimension, thus can support
/// physics systems with any dimensions, 1D, 2D, 3D, 4D, ...

//
//
//
//
//
#include "ARIA/ForEach.h"
#include "ARIA/Math.h"
#include "ARIA/Vec.h"

namespace ARIA {

/// \brief An axis-aligned bounding box (AABB) implementation.
///
/// \tparam d Dimension.
///
/// \example ```cpp
/// // Create an `AABB`.
/// AABB3r aabb{Vec3r{0, 4, 2}, Vec3r{3, 1, 5}};
///
/// // Get the infimum and supremum of the `AABB`.
/// // See `Math.h` for the definition of `supremum` and `infimum`.
/// Vec3r inf = aabb.inf();
/// Vec3r sup = aabb.sup();
/// ```
///
/// \note `AABB` is implemented with `Vec`, which internally uses `Eigen::Vector`.
/// So, all functions are currently not `constexpr`.
template <typename T, uint d>
class AABB final {
public:
  /// \brief Construct an empty `AABB`, whose each dimension `i`,
  /// `inf()[i]` equals to `supremum<T>`, while `sup()[i]` equals to `infimum<T>`.
  /// See `Math.h` for the definition of `supremum` and `infimum`.
  ///
  /// \example ```cpp
  /// AABB3r aabb;
  /// ```
  ARIA_HOST_DEVICE inline /*constexpr*/ AABB();

  /// \brief Construct a non-empty `AABB` with the given `args`.
  /// `args` can be a combination of `AABB`s and `Vec`s, where `Vec`s indicates solo points.
  /// Then, this `AABB` is constructed with the union of all the `args`.
  ///
  /// \example ```cpp
  /// AABB3r anotherAABB = ...;
  /// Vec3r point{1, 2, 3};
  ///
  /// AABB3r aabb0{anotherAABB, point};
  /// AABB3r aabb1{point, anotherAABB}; // The same.
  /// ```
  ///
  /// \warning Please read the comments of `empty`.
  /// Any `AABB` constructed with this constructor is guaranteed to be non-empty.
  template <typename... Args>
    requires(sizeof...(Args) > 0 &&  // Size `> 0` to avoid conflict with the default constructor.
             (sizeof...(Args) > 1 || // If size `> 1`, safe. If size `== 1`, may conflict with the copy constructor.
              (!std::is_same_v<std::decay_t<Args>, AABB> && ...))) // So, requires that the argument type is not AABB.
  ARIA_HOST_DEVICE /*constexpr*/ explicit AABB(Args &&...args) : AABB(unionized(std::forward<Args>(args)...)) {}

  /// \brief `AABB` allows copy and move.
  ARIA_COPY_MOVE_ABILITY(AABB, default, default);

  ~AABB() = default;

  //
  //
  //
public:
  /// \brief The infimum of the `AABB`.
  /// You may imagine that `inf` is the position of the "left-bottom" corner of the `AABB`.
  ///
  /// \example ```cpp
  /// AABB3r aabb;
  /// Vec3r inf = aabb.inf();       // Get the `inf`.
  /// aabb.inf() = {1_R, 2_R, 3_R}; // Set the `inf`.
  /// ```
  ///
  /// \warning You are allowed to directly set the `inf`,
  /// if some dimension of the new `inf` is larger than the `sup`,
  /// the `AABB` will become an `empty` `AABB`.
  /// So, pay attention if you want to directly set `inf`.
  ///
  /// You may use `Unionize()` if you always want a non-empty `AABB`.
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, inf, infAndSup_[0]);

  /// \brief The supremum of the `AABB`.
  /// You may imagine that `sup` is the position of the "right-top" corner of the `AABB`.
  ///
  /// \example ```cpp
  /// AABB3r aabb;
  /// Vec3r sup = aabb.sup();       // Get the `sup`.
  /// aabb.sup() = {1_R, 2_R, 3_R}; // Set the `sup`.
  /// ```
  ///
  /// \warning You are allowed to directly set the `sup`,
  /// if some dimension of the new `sup` is smaller than the `inf`,
  /// the `AABB` will become an `empty` `AABB`.
  /// So, pay attention if you want to directly set `sup`.
  ///
  /// You may use `Unionize()` if you always want a non-empty `AABB`.
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, sup, infAndSup_[1]);

public:
  /// \brief Access the infimum or the supremum of the `AABB` with an index.
  /// Infimum is returned if `i == 0`, supremum is returned if `i == 1`.
  /// Indices besides 0 or 1 are invalid.
  ///
  /// \example ```cpp
  /// Vec3r inf0 = aabb[0]; // Get the `inf`.
  /// Vec3r sup0 = aabb[1]; // Get the `sup`.
  /// ```
  [[nodiscard]] ARIA_HOST_DEVICE inline /*constexpr*/ const Vec<T, d> &operator[](uint i) const;

  /// \brief Access the infimum or the supremum of the `AABB` with an index.
  /// Infimum is returned if `i == 0`, supremum is returned if `i == 1`.
  /// Indices besides 0 or 1 are invalid.
  ///
  /// \example ```cpp
  /// Vec3r inf0 = aabb[0]; // Get the `inf`.
  /// Vec3r sup0 = aabb[1]; // Get the `sup`.
  /// ```
  [[nodiscard]] ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> &operator[](uint i);

  //
  //
  //
public:
  /// \brief An `AABB` is defined as empty if there exist one dimension `i`,
  /// such that `inf()[i] > sup()[i]`.
  ///
  /// \warning If the `AABB` is constructed with only one point,
  /// by definition, it is also considered as non-empty.
  [[nodiscard]] ARIA_HOST_DEVICE inline /*constexpr*/ bool empty() const;

  /// \brief Unionize this `AABB` with the given `args`.
  /// Equivalent to `thisAABB = AABB{thisAABB, ... /* Args here. */};
  template <typename... Args>
  ARIA_HOST_DEVICE inline /*constexpr*/ void Unionize(Args &&...args);

  //
  //
  //
public:
  /// \brief `diagonal` of an `AABB` is defined as
  /// `sup() - inf()`.
  [[nodiscard]] ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> diagonal() const;

  /// \brief `center` of an `AABB` is defined as
  /// `(inf() + sup()) / 2`.
  [[nodiscard]] ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> center() const;

  /// \brief `offset` of a given point in an `AABB` is defined as
  /// `(p - inf()).cwiseQuotient(sup() - inf())`.
  [[nodiscard]] ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> offset(const Vec<T, d> &p) const;

  //
  //
  //
  //
  //
private:
  std::array<Vec<T, d>, 2> infAndSup_;

  // A supporting function to help implement the constructor and `Unionize()`.
  template <typename... Args>
  [[nodiscard]] ARIA_HOST_DEVICE static inline /*constexpr*/ AABB unionized(Args &&...args);
};

//
//
//
//
//
template <typename T>
using AABB1 = AABB<T, 1>;
template <typename T>
using AABB2 = AABB<T, 2>;
template <typename T>
using AABB3 = AABB<T, 3>;
template <typename T>
using AABB4 = AABB<T, 4>;

using AABB1i = AABB1<int>;
using AABB1u = AABB1<uint>;
using AABB1f = AABB1<float>;
using AABB1d = AABB1<double>;
using AABB1r = AABB1<Real>;

using AABB2i = AABB2<int>;
using AABB2u = AABB2<uint>;
using AABB2f = AABB2<float>;
using AABB2d = AABB2<double>;
using AABB2r = AABB2<Real>;

using AABB3i = AABB3<int>;
using AABB3u = AABB3<uint>;
using AABB3f = AABB3<float>;
using AABB3d = AABB3<double>;
using AABB3r = AABB3<Real>;

using AABB4i = AABB4<int>;
using AABB4u = AABB4<uint>;
using AABB4f = AABB4<float>;
using AABB4d = AABB4<double>;
using AABB4r = AABB4<Real>;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/AABB.inc"
