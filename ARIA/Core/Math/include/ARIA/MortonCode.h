#pragma once

#include "ARIA/Vec.h"

/// \details The Morton code implementations are based on:
/// 1. http://www.graphics.stanford.edu/~seander/bithacks.html,
/// 2. https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits,
/// 3. https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints.

#if 0
// 3D 32
x &= 0x3ff
x = (x | x << 16) & 0x30000ff   #<<< THIS IS THE MASK for shifting 16 (for bit 8 and 9)
x = (x | x << 8) & 0x300f00f
x = (x | x << 4) & 0x30c30c3
x = (x | x << 2) & 0x9249249




// 3D 64
x &= 0x1fffff
x = (x | x << 32) & 0x1f00000000ffff
x = (x | x << 16) & 0x1f0000ff0000ff
x = (x | x << 8) & 0x100f00f00f00f00f
x = (x | x << 4) & 0x10c30c30c30c30c3
x = (x | x << 2) & 0x1249249249249249




// 3D 128
x &= 0x3ffffffffff
x = (x | x << 64) & 0x3ff0000000000000000ffffffffL
x = (x | x << 32) & 0x3ff00000000ffff00000000ffffL
x = (x | x << 16) & 0x30000ff0000ff0000ff0000ff0000ffL
x = (x | x << 8) & 0x300f00f00f00f00f00f00f00f00f00fL
x = (x | x << 4) & 0x30c30c30c30c30c30c30c30c30c30c3L
x = (x | x << 2) & 0x9249249249249249249249249249249L
#endif

namespace ARIA {

template <std::integral I>
[[nodiscard]] ARIA_HOST_DEVICE I MortonCode(const Vec2<I> &coord) {
  if constexpr (sizeof(I) == 4) {
    auto shift = [](I x) {
      x &= I{0xFFFF};
      x = (x | (x << 8)) & I{0x00FF00FF};
      x = (x | (x << 4)) & I{0x0F0F0F0F};
      x = (x | (x << 2)) & I{0x33333333};
      x = (x | (x << 1)) & I{0x55555555};
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
  }
}

} // namespace ARIA
