#pragma once

#include "ARIA/Property.h"

#include <Eigen/Core>

namespace ARIA {

namespace mat::detail {

//! For future developers, never wrap `Eigen` types with composition or inheritance, or
//! everything will be extremely complicated.
//!
//! For example if you write something like this:
//!   template <typename T, auto row, auto col>
//!   class Mat : public Eigen::Matrix<T, row, col> { ... };
//! and:
//!   template <typename T, auto size>
//!   class Vec : public Eigen::Vector<T, s> { ... };
//!
//! Then, your `Mat` and `Vec` have different types!
//! But actually, `Eigen::Vector` and `Eigen::Matrix` share the same type.
//!
//! This is the simplest drawback, things will be much more weird
//! when you call some `Eigen` functions.
//! So, never warp `Eigen`, directly use it.
template <typename T, auto row, auto col>
using Mat = Eigen::Matrix<T, row, col>;

//
//
//
// Whether the given type is `Mat<T, ...>`.
template <typename T>
struct is_mat : std::false_type {};

template <typename T, auto row, auto col>
struct is_mat<Mat<T, row, col>> : std::true_type {};

template <typename T>
static constexpr bool is_mat_v = is_mat<T>::value;

//
//
//
// Whether the given type is `Mat<T, row, col>.
template <typename T, auto row, auto col>
struct is_mat_rc : std::false_type {};

template <typename T, auto row, auto col>
struct is_mat_rc<Mat<T, row, col>, row, col> : std::true_type {};

template <typename T, auto row, auto col>
static constexpr bool is_mat_rc_v = is_mat_rc<T, row, col>::value;

//
//
//
//
//
#define __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_MAT(specifiers, type)                                                  \
                                                                                                                       \
  /* A. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., adjoint);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., adjointInPlace);                                                               \
  ARIA_PROP_FUNC(public, specifiers, ., all);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., allFinite);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., any);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., array);                                                                        \
  /* B. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., begin);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., block);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., bottomLeftCorner);                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., bottomRightCorner);                                                            \
  ARIA_PROP_FUNC(public, specifiers, ., bottomRows);                                                                   \
  /* C. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., cast);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., cbegin);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., cend);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., coeff);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., coeffRef);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., col);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., cols);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., colwise);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., conjugate);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., conjugateIf);                                                                  \
  ARIA_PROP_FUNC(public, specifiers, ., cross);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., cross3);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseAbs);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseAbs2);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseArg);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseEqual);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseInverse);                                                                 \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseMax);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseMin);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseNotEqual);                                                                \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseProduct);                                                                 \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseQuotient);                                                                \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseSign);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., cwiseSqrt);                                                                    \
  /* D. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., data);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., determinant);                                                                  \
  ARIA_PROP_FUNC(public, specifiers, ., diagonal);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., diagonalSize);                                                                 \
  ARIA_PROP_FUNC(public, specifiers, ., dot);                                                                          \
  /* E. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., eigenvalues);                                                                  \
  ARIA_PROP_FUNC(public, specifiers, ., end);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., eulerAngles);                                                                  \
  ARIA_PROP_FUNC(public, specifiers, ., eval);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., evalTo);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., exp);                                                                          \
  /* F. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., fill);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., format);                                                                       \
  /* G. */                                                                                                             \
  /* H. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., hasNaN);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., homogeneous);                                                                  \
  /* I. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., inverse);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., isApprox);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., isApproxToConstant);                                                           \
  ARIA_PROP_FUNC(public, specifiers, ., isConstant);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., isDiagonal);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., isIdentity);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., isLowerTriangular);                                                            \
  ARIA_PROP_FUNC(public, specifiers, ., isMuchSmallerThan);                                                            \
  ARIA_PROP_FUNC(public, specifiers, ., isOnes);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., isOrthogonal);                                                                 \
  ARIA_PROP_FUNC(public, specifiers, ., isUnitary);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., isUpperTriangular);                                                            \
  ARIA_PROP_FUNC(public, specifiers, ., isZero);                                                                       \
  /* J. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., jacobiSvd);                                                                    \
  /* K. */                                                                                                             \
  /* L. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., leftCols);                                                                     \
  /* M. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., matrix);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., maxCoeff);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., mean);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., middleCols);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., middleRows);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., minCoeff);                                                                     \
  /* N. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., noalias);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., nonZeros);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., norm);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., normalize);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., normalized);                                                                   \
  /* O. */                                                                                                             \
  /* P. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., packet);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., pow);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., prod);                                                                         \
  /* Q. */                                                                                                             \
  /* R. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., real);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., redux);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., replicate);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., reshaped);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., resize);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., resizeLike);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., reverse);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., reverseInPlace);                                                               \
  ARIA_PROP_FUNC(public, specifiers, ., rightCols);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., row);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., rows);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., rowwise);                                                                      \
  /* S. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., segment);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., select);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., setConstant);                                                                  \
  ARIA_PROP_FUNC(public, specifiers, ., setIdentity);                                                                  \
  ARIA_PROP_FUNC(public, specifiers, ., setOnes);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., setRandom);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., setUnit);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., setZero);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., sin);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., sinh);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., size);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., sparseView);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., sqrt);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., squaredNorm);                                                                  \
  ARIA_PROP_FUNC(public, specifiers, ., stableNorm);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., stableNormalize);                                                              \
  ARIA_PROP_FUNC(public, specifiers, ., stableNormalized);                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., subVector);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., subVectors);                                                                   \
  ARIA_PROP_FUNC(public, specifiers, ., sum);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., swap);                                                                         \
  /* T. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., tail);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., topLeftCorner);                                                                \
  ARIA_PROP_FUNC(public, specifiers, ., topRightCorner);                                                               \
  ARIA_PROP_FUNC(public, specifiers, ., topRows);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., trace);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., transpose);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., transposeInPlace);                                                             \
  /* U. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., unitOrthogonal);                                                               \
  /* V. */                                                                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., value);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., visit);                                                                        \
  /* W. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, w);                                                            \
  ARIA_PROP_FUNC(public, specifiers, ., writePacket);                                                                  \
  /* X. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, x);                                                            \
  /* Y. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, y);                                                            \
  /* Z. */                                                                                                             \
  ARIA_SUB_PROP(specifiers, std::decay_t<type>::Scalar, z);

#define __ARIA_PROP_PREFAB_MAT(accessGet, accessSet, specifiers, type, propName)                                       \
  static_assert(ARIA::mat::detail::is_mat_v<std::decay_t<type>>,                                                       \
                "Type of the property should be `class Mat` in order to use this prefab");                             \
                                                                                                                       \
  ARIA_PROP_BEGIN(accessGet, accessSet, specifiers, type, propName);                                                   \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_MAT(specifiers, type);                                                       \
  ARIA_PROP_END

#define __ARIA_SUB_PROP_PREFAB_MAT(specifiers, type, propName)                                                         \
  static_assert(ARIA::mat::detail::is_mat_v<std::decay_t<type>>,                                                       \
                "Type of the property should be `class Mat` in order to use this prefab");                             \
                                                                                                                       \
  ARIA_SUB_PROP_BEGIN(specifiers, type, propName);                                                                     \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_MAT(specifiers, type);                                                       \
  ARIA_PROP_END

} // namespace mat::detail

} // namespace ARIA
