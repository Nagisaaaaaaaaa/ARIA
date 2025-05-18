#pragma once

#include "ARIA/Mosaic.h"
#include "ARIA/Property.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ARIA {

namespace quat::detail {

// Similar to `Mat`.
template <typename T>
using Quat = Eigen::Quaternion<T>;

//
//
//
// Whether the given type is `Quat<T>`.
template <typename T>
struct is_quat : std::false_type {};

template <typename T>
struct is_quat<Quat<T>> : std::true_type {};

template <typename T>
static constexpr bool is_quat_v = is_quat<T>::value;

//
//
//
//
//
#define __ARIA_PROP_AND_SUB_PROP_INCOMPLETE_PREFAB_MEMBERS_QUAT(specifiers, type)                                      \
                                                                                                                       \
  /* W. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, w);                                                            \
  /* X. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, x);                                                            \
  /* Y. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, y);                                                            \
  /* Z. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, z);

#define __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_QUAT(specifiers, type)                                                 \
                                                                                                                       \
  /* A. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., angularDistance);                                                              \
  /* B. */                                                                                                             \
  /* C. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., cast);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., coeffs);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., conjugate);                                                                    \
  /* D. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., dot);                                                                          \
  /* E. */                                                                                                             \
  /* F. */                                                                                                             \
  /* G. */                                                                                                             \
  /* H. */                                                                                                             \
  /* I. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., inverse);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., isApprox);                                                                     \
  /* J. */                                                                                                             \
  /* K. */                                                                                                             \
  /* L. */                                                                                                             \
  /* M. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., matrix);                                                                       \
  /* N. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., norm);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., normalize);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., normalized);                                                                   \
  /* O. */                                                                                                             \
  /* P. */                                                                                                             \
  /* Q. */                                                                                                             \
  /* R. */                                                                                                             \
  /* S. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., setFromTwoVectors);                                                            \
  ARIA_PROP_FUNC(public, specifiers, ., setIdentity);                                                                  \
  ARIA_PROP_FUNC(public, specifiers, ., slerp);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., squaredNorm);                                                                  \
  /* T. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., toRotationMatrix);                                                             \
  /* U. */                                                                                                             \
  /* V. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., vec);                                                                          \
  /* W. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, w);                                                            \
  /* X. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, x);                                                            \
  /* Y. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, y);                                                            \
  /* Z. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, z);

//
//
//
#define __ARIA_PROP_INCOMPLETE_PREFAB_QUAT(accessGet, accessSet, specifiers, type, /*propName,*/...)                   \
  static_assert(ARIA::quat::detail::is_quat_v<std::decay_t<type>>,                                                     \
                "Type of the property should be `class Quat` in order to use this prefab");                            \
                                                                                                                       \
  ARIA_PROP_BEGIN(accessGet, accessSet, specifiers, type, /*propName,*/ __VA_ARGS__);                                  \
  __ARIA_PROP_AND_SUB_PROP_INCOMPLETE_PREFAB_MEMBERS_QUAT(specifiers, type);                                           \
  ARIA_PROP_END

#define __ARIA_SUB_PROP_INCOMPLETE_PREFAB_QUAT(specifiers, type, /*propName,*/...)                                     \
  static_assert(ARIA::quat::detail::is_quat_v<std::decay_t<type>>,                                                     \
                "Type of the property should be `class Quat` in order to use this prefab");                            \
                                                                                                                       \
  ARIA_SUB_PROP_BEGIN(specifiers, type, /*propName,*/ __VA_ARGS__);                                                    \
  __ARIA_PROP_AND_SUB_PROP_INCOMPLETE_PREFAB_MEMBERS_QUAT(specifiers, type);                                           \
  ARIA_PROP_END

#define __ARIA_PROP_PREFAB_QUAT(accessGet, accessSet, specifiers, type, /*propName,*/...)                              \
  static_assert(ARIA::quat::detail::is_quat_v<std::decay_t<type>>,                                                     \
                "Type of the property should be `class Quat` in order to use this prefab");                            \
                                                                                                                       \
  ARIA_PROP_BEGIN(accessGet, accessSet, specifiers, type, /*propName,*/ __VA_ARGS__);                                  \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_QUAT(specifiers, type);                                                      \
  ARIA_PROP_END

#define __ARIA_SUB_PROP_PREFAB_QUAT(specifiers, type, /*propName,*/...)                                                \
  static_assert(ARIA::quat::detail::is_quat_v<std::decay_t<type>>,                                                     \
                "Type of the property should be `class Quat` in order to use this prefab");                            \
                                                                                                                       \
  ARIA_SUB_PROP_BEGIN(specifiers, type, /*propName,*/ __VA_ARGS__);                                                    \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_QUAT(specifiers, type);                                                      \
  ARIA_PROP_END

//
//
//
//
//
// Define the built-in `Mosaic` for `Quat`.
template <typename T>
struct MosaicPatternQuat {
  T v[4];
};

//
//
//
template <typename T>
struct reduce_quat_mosaic;

// 1. `QuatMosaic<T>`.
template <typename T>
  requires(!is_quat_v<T>)
struct reduce_quat_mosaic<T> {
  using type = Mosaic<Quat<T>, MosaicPatternQuat<T>>;
};

// 2. `QuatMosaic<Quat<T>>`.
template <typename T>
  requires(is_quat_v<Quat<T>>)
struct reduce_quat_mosaic<Quat<T>> {
  using type = Mosaic<Quat<T>, MosaicPatternQuat<T>>;
};

template <typename T>
using reduce_quat_mosaic_t = typename reduce_quat_mosaic<T>::type;

} // namespace quat::detail

//
//
//
template <typename T>
class Mosaic<quat::detail::Quat<T>, quat::detail::MosaicPatternQuat<T>> {
private:
  using TValue = quat::detail::Quat<T>;
  using TPattern = quat::detail::MosaicPatternQuat<T>;

public:
  [[nodiscard]] TPattern operator()(const TValue &value) const {
    return {.v = {value.w(), value.x(), value.y(), value.z()}};
  }

  [[nodiscard]] TValue operator()(const TPattern &pattern) const {
    return {pattern.v[0], pattern.v[1], pattern.v[2], pattern.v[3]};
  }
};

} // namespace ARIA
