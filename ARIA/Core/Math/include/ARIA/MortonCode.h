#pragma once

/// \file
/// \brief

#include "ARIA/Vec.h"

namespace ARIA {

/// \example ```cpp
/// using Code2D = MortonCode<2>;
/// using Code3D = MortonCode<3>;
///
/// uint code = Code2D::Encode(Vec2u{5, 6});
/// uint code = Code3D::Encode(Vec3u{5, 6, 7});
/// ```
template <uint d>
class MortonCode;

template <>
class MortonCode<2> {
public:
  /// \brief 2D Morton code encoding function.
  ///
  /// \example ```cpp
  /// using Code = MortonCode<2>;
  ///
  /// uint code = Code::Encode(Vec2u{5, 6});
  /// ```
  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE /*constexpr*/ I Encode(const Vec2<I> &coord);
};

template <>
class MortonCode<3> {
public:
  /// \brief 3D Morton code encoding function.
  ///
  /// \example ```cpp
  /// using Code = MortonCode<3>;
  ///
  /// uint code = Code::Encode(Vec3u{5, 6, 7});
  /// ```
  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE /*constexpr*/ I Encode(const Vec3<I> &coord);
};

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/MortonCode.inc"
