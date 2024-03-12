#pragma once

/// \file
/// \brief Morton curve, or the so-called z-order curve, is a space filling curve.
/// A space filling curve maps multidimensional data to one dimension while
/// preserving locality of the data points.
/// See https://en.wikipedia.org/wiki/Z-order_curve.
///
/// This file introduces a Morton code encoder and decoder, which can be used to
/// encode a N-dimensional integral coordinate to an integer, or
/// decode an integer to a N-dimensional integral coordinate.

//
//
//
//
//
#include "ARIA/Vec.h"

namespace ARIA {

/// \brief A Morton code encoder and decoder, which can be used to
/// encode a N-dimensional integral coordinate to an integer, or
/// decode an integer to a N-dimensional integral coordinate.
/// See https://en.wikipedia.org/wiki/Z-order_curve.
///
/// \tparam d Dimension.
///
/// \example ```cpp
/// // 1D.
/// using Code = MortonCode<1>;
///
/// uint code0 = Code::Encode(Vec1u{0}); // Will be 0.
/// uint code1 = Code::Encode(Vec1u{1}); //         1.
/// uint code2 = Code::Encode(Vec1u{2}); //         2.
/// uint code2 = Code::Encode(Vec1u{3}); //         3.
///
/// Vec1u coord0 = Code::Decode(0u); // Will be {0}.
/// Vec1u coord1 = Code::Decode(1u); //         {1}.
/// Vec1u coord2 = Code::Decode(2u); //         {2}.
/// Vec1u coord2 = Code::Decode(3u); //         {3}.
///
/// // 2D.
/// using Code = MortonCode<2>;
///
/// uint code0 = Code::Encode(Vec2u{0, 0}); // Will be 0.
/// uint code1 = Code::Encode(Vec2u{1, 0}); //         1.
/// uint code2 = Code::Encode(Vec2u{0, 1}); //         2.
/// uint code3 = Code::Encode(Vec2u{1, 1}); //         3.
///
/// Vec2u coord0 = Code::Decode(0u); // Will be {0, 0}.
/// Vec2u coord1 = Code::Decode(1u); //         {1, 0}.
/// Vec2u coord2 = Code::Decode(2u); //         {0, 1}.
/// Vec2u coord3 = Code::Decode(3u); //         {1, 1}.
///
/// // 3D.
/// using Code = MortonCode<3>;
///
/// uint code0 = Code::Encode(Vec3u{0, 0, 0}); // Will be 0.
/// uint code1 = Code::Encode(Vec3u{1, 0, 0}); //         1.
/// uint code2 = Code::Encode(Vec3u{0, 1, 0}); //         2.
/// uint code3 = Code::Encode(Vec3u{1, 1, 0}); //         3.
/// uint code4 = Code::Encode(Vec3u{0, 0, 1}); //         4.
/// uint code5 = Code::Encode(Vec3u{1, 0, 1}); //         5.
/// uint code6 = Code::Encode(Vec3u{0, 1, 1}); //         6.
/// uint code7 = Code::Encode(Vec3u{1, 1, 1}); //         7.
///
/// Vec3u coord0 = Code::Decode(0u); // Will be {0, 0, 0}.
/// Vec3u coord1 = Code::Decode(1u); //         {1, 0, 0}.
/// Vec3u coord2 = Code::Decode(2u); //         {0, 1, 0}.
/// Vec3u coord3 = Code::Decode(3u); //         {1, 1, 0}.
/// Vec3u coord4 = Code::Decode(4u); //         {0, 0, 1}.
/// Vec3u coord5 = Code::Decode(5u); //         {1, 0, 1}.
/// Vec3u coord6 = Code::Decode(6u); //         {0, 1, 1}.
/// Vec3u coord7 = Code::Decode(7u); //         {1, 1, 1}.
/// ```
template <auto d>
class MortonCode;

//
//
//
//
//
// 1D specialization.
template <>
class MortonCode<1> {
public:
  /// \brief Encode a 1D coordinate to Morton code.
  ///
  /// \example ```cpp
  /// using Code = MortonCode<1>; // 1D.
  ///
  /// uint code0 = Code::Encode(Vec1u{0}); // Will be 0.
  /// uint code1 = Code::Encode(Vec1u{1}); //         1.
  /// uint code2 = Code::Encode(Vec1u{2}); //         2.
  /// uint code3 = Code::Encode(Vec1u{3}); //         3.
  /// ```
  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE constexpr I Encode(const Vec1<I> &coord) {
    return coord.x();
  }

  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE constexpr Vec1<I> Decode(const I &code) {
    return Vec1<I>{code};
  }
};

//
//
//
// 2D specialization.
template <>
class MortonCode<2> {
public:
  /// \brief Encode a 2D coordinate to Morton code.
  ///
  /// \example ```cpp
  /// using Code = MortonCode<2>; // 2D.
  ///
  /// uint code0 = Code::Encode(Vec2u{0, 0}); // Will be 0.
  /// uint code1 = Code::Encode(Vec2u{1, 0}); //         1.
  /// uint code2 = Code::Encode(Vec2u{0, 1}); //         2.
  /// uint code3 = Code::Encode(Vec2u{1, 1}); //         3.
  /// ```
  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE inline constexpr I Encode(const Vec2<I> &coord);

  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE inline constexpr Vec2<I> Decode(const I &code);
};

//
//
//
// 3D specialization.
template <>
class MortonCode<3> {
public:
  /// \brief Encode a 3D coordinate to Morton code.
  ///
  /// \example ```cpp
  /// using Code = MortonCode<3>; // 3D.
  ///
  /// uint code0 = Code::Encode(Vec3u{0, 0, 0}); // Will be 0.
  /// uint code1 = Code::Encode(Vec3u{1, 0, 0}); //         1.
  /// uint code2 = Code::Encode(Vec3u{0, 1, 0}); //         2.
  /// uint code3 = Code::Encode(Vec3u{1, 1, 0}); //         3.
  /// uint code4 = Code::Encode(Vec3u{0, 0, 1}); //         4.
  /// uint code5 = Code::Encode(Vec3u{1, 0, 1}); //         5.
  /// uint code6 = Code::Encode(Vec3u{0, 1, 1}); //         6.
  /// uint code7 = Code::Encode(Vec3u{1, 1, 1}); //         7.
  /// ```
  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE inline constexpr I Encode(const Vec3<I> &coord);

  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE inline constexpr Vec3<I> Decode(const I &code);
};

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/MortonCode.inc"
