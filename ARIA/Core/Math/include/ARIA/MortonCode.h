#pragma once

/// \file
/// \details The Morton code implementations are based on:
/// 1. http://www.graphics.stanford.edu/~seander/bithacks.html,
/// 2. https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits,
/// 3. https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints.

#include "ARIA/Vec.h"

namespace ARIA {

/// \example ```cpp
/// using Code = MortonCode;
///
/// uint code = Code::Encode(Vec2u{5, 6});
/// uint code = Code::Encode(Vec3u{5, 6, 7});
/// ```
class MortonCode {
public:
  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE /*constexpr*/ I Encode(const Vec2<I> &coord) {
    if constexpr (sizeof(I) == 4) {
      auto shift = [](I x) {
        x &= I{0xFFFFU};
        x = (x | (x << 8)) & I{0x00FF00FFU};
        x = (x | (x << 4)) & I{0x0F0F0F0FU};
        x = (x | (x << 2)) & I{0x33333333U};
        x = (x | (x << 1)) & I{0x55555555U};
        return x;
      };
      return shift(coord.x()) | (shift(coord.y()) << 1);
    } else if constexpr (sizeof(I) == 8) {
      auto shift = [](I x) {
        x &= I{0xFFFFFFFFLLU};
        x = (x | (x << 16)) & I{0x0000FFFF0000FFFFLLU};
        x = (x | (x << 8)) & I{0x00FF00FF00FF00FFLLU};
        x = (x | (x << 4)) & I{0x0F0F0F0F0F0F0F0FLLU};
        x = (x | (x << 2)) & I{0x3333333333333333LLU};
        x = (x | (x << 1)) & I{0x5555555555555555LLU};
        return x;
      };
      return shift(coord.x()) | (shift(coord.y()) << 1);
    } else
      ARIA_STATIC_ASSERT_FALSE("Type of the coord elements to be encoded is currently not supported");
  }

  template <std::integral I>
  [[nodiscard]] static ARIA_HOST_DEVICE /*constexpr*/ I Encode(const Vec3<I> &coord) {
    if constexpr (sizeof(I) == 4) {
      auto shift = [](I x) {
        x &= I{0x3FFU};
        x = (x | x << 16) & I{0x30000FFU};
        x = (x | x << 8) & I{0x300F00FU};
        x = (x | x << 4) & I{0x30C30C3U};
        x = (x | x << 2) & I{0x9249249U};
        return x;
      };
      return shift(coord.x()) | (shift(coord.y()) << 1) | (shift(coord.z()) << 2);
    } else if constexpr (sizeof(I) == 8) {
      auto shift = [](I x) {
        x &= I{0x1FFFFFLLU};
        x = (x | x << 32) & I{0x1F00000000FFFFLLU};
        x = (x | x << 16) & I{0x1F0000FF0000FFLLU};
        x = (x | x << 8) & I{0x100F00F00F00F00FLLU};
        x = (x | x << 4) & I{0x10C30C30C30C30C3LLU};
        x = (x | x << 2) & I{0x1249249249249249LLU};
        return x;
      };
      return shift(coord.x()) | (shift(coord.y()) << 1) | (shift(coord.z()) << 2);
    } /* else if constexpr (sizeof(I) == 16) {
       auto shift = [](I x) {
         x &= I{0X3FFFFFFFFFFLLLU};
         x = (x | x << 64) & I{0x3FF0000000000000000FFFFFFFFLLLU};
         x = (x | x << 32) & I{0x3FF00000000FFFF00000000FFFFLLLU};
         x = (x | x << 16) & I{0x30000FF0000FF0000FF0000FF0000FFLLLU};
         x = (x | x << 8) & I{0x300F00F00F00F00F00F00F00F00F00FLLLU};
         x = (x | x << 4) & I{0x30C30C30C30C30C30C30C30C30C30C3LLLU};
         x = (x | x << 2) & I{0x9249249249249249249249249249249LLLU};
         return x;
       };
       return shift(coord.x()) | (shift(coord.y()) << 1) | (shift(coord.z()) << 2);
     }*/
    else
      ARIA_STATIC_ASSERT_FALSE("Type of the coord elements to be encoded is currently not supported");
  }
};

} // namespace ARIA
