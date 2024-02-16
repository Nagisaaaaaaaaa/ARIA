#pragma once

#include "ARIA/ForEach.h"
#include "ARIA/Math.h"
#include "ARIA/Vec.h"

namespace ARIA {

/// \note `AABB` is implemented with `Vec`, which internally uses `Eigen::Vector`.
/// So, all functions are currently not `constexpr`.
template <typename T, auto d>
class AABB final {
public:
  /// \brief Construct an empty `AABB`, whose each dimension `i`,
  /// `inf()[i]` equals to `supremum<T>`, while `sup()[i]` equals to `infimum<T>`.
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
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, inf, infAndSup_[0]);
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, sup, infAndSup_[1]);

public:
  ARIA_HOST_DEVICE inline /*constexpr*/ const Vec<T, d> &operator[](uint i) const;

  ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> &operator[](uint i);

  //
  //
  //
public:
  /// \brief An `AABB` is defined as empty if there exist one dimension `i`,
  /// such that `inf()[i] > sup()[i]`.
  ///
  /// \warning If the `AABB` is constructed with only one point,
  /// by definition, it is also considered as non-empty.
  ARIA_HOST_DEVICE inline /*constexpr*/ bool empty() const;

  template <typename... Args>
  ARIA_HOST_DEVICE static inline /*constexpr*/ AABB unionized(Args &&...args);

  template <typename... Args>
  ARIA_HOST_DEVICE inline /*constexpr*/ void Unionize(Args &&...args);

  //
  //
  //
public:
  ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> center() const;

  ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> offset(const Vec<T, d> &p) const;

  ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> diagonal() const;

  //
  //
  //
private:
  std::array<Vec<T, d>, 2> infAndSup_;
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
