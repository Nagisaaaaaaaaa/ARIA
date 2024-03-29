#pragma once

namespace ARIA {

//! The Morton code implementations are based on bit hacks, see:
//! 1. http://www.graphics.stanford.edu/~seander/bithacks.html,
//! 2. https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits,
//! 3. https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints.
//! 4. Feed the encoders to ChatGPT and it will tell you how to implement the decoders.
//!
//! Lookup Table (LUT) method is not used to make it easier to work with CUDA.

//
//
//
//
//
template <std::integral I>
[[nodiscard]] ARIA_HOST_DEVICE inline constexpr I MortonCode<2>::Encode(const Vec2<I> &coord) {
  auto shift = [](I x) {
    if constexpr (sizeof(I) == 4) {
      x &= I{0xFFFFU};
      x = (x | (x << 8)) & I{0x00FF00FFU};
      x = (x | (x << 4)) & I{0x0F0F0F0FU};
      x = (x | (x << 2)) & I{0x33333333U};
      x = (x | (x << 1)) & I{0x55555555U};
      return x;
    } else if constexpr (sizeof(I) == 8) {
      x &= I{0xFFFFFFFFLLU};
      x = (x | (x << 16)) & I{0x0000FFFF0000FFFFLLU};
      x = (x | (x << 8)) & I{0x00FF00FF00FF00FFLLU};
      x = (x | (x << 4)) & I{0x0F0F0F0F0F0F0F0FLLU};
      x = (x | (x << 2)) & I{0x3333333333333333LLU};
      x = (x | (x << 1)) & I{0x5555555555555555LLU};
      return x;
    } else
      ARIA_STATIC_ASSERT_FALSE("Type of the coord elements to be encoded is currently not supported");
  };

  return shift(coord.x()) | (shift(coord.y()) << 1);
}

template <std::integral I>
[[nodiscard]] ARIA_HOST_DEVICE inline constexpr Vec2<I> MortonCode<2>::Decode(const I &code) {
  auto unshift = [](I x) {
    if constexpr (sizeof(I) == 4) {
      x &= I{0x55555555U};
      x = (x ^ (x >> 1)) & I{0x33333333U};
      x = (x ^ (x >> 2)) & I{0x0F0F0F0FU};
      x = (x ^ (x >> 4)) & I{0x00FF00FFU};
      x = (x ^ (x >> 8)) & I{0x0000FFFFU};
      return x;
    } else if constexpr (sizeof(I) == 8) {
      x &= I{0x5555555555555555LLU};
      x = (x ^ (x >> 1)) & I{0x3333333333333333LLU};
      x = (x ^ (x >> 2)) & I{0x0F0F0F0F0F0F0F0FLLU};
      x = (x ^ (x >> 4)) & I{0x00FF00FF00FF00FFLLU};
      x = (x ^ (x >> 8)) & I{0x0000FFFF0000FFFFLLU};
      x = (x ^ (x >> 16)) & I{0x00000000FFFFFFFFLLU};
      return x;
    }
  };

  return {unshift(code), unshift(code >> 1)};
}

//
//
//
//
//
template <std::integral I>
[[nodiscard]] ARIA_HOST_DEVICE inline constexpr I MortonCode<3>::Encode(const Vec3<I> &coord) {
  auto shift = [](I x) {
    if constexpr (sizeof(I) == 4) {
      x &= I{0x3FFU};
      x = (x | x << 16) & I{0x30000FFU};
      x = (x | x << 8) & I{0x300F00FU};
      x = (x | x << 4) & I{0x30C30C3U};
      x = (x | x << 2) & I{0x9249249U};
      return x;
    } else if constexpr (sizeof(I) == 8) {
      x &= I{0x1FFFFFLLU};
      x = (x | x << 32) & I{0x1F00000000FFFFLLU};
      x = (x | x << 16) & I{0x1F0000FF0000FFLLU};
      x = (x | x << 8) & I{0x100F00F00F00F00FLLU};
      x = (x | x << 4) & I{0x10C30C30C30C30C3LLU};
      x = (x | x << 2) & I{0x1249249249249249LLU};
      return x;
    } /*else if constexpr (sizeof(I) == 16) {
      x &= I{0X3FFFFFFFFFFLLLU};
      x = (x | x << 64) & I{0x3FF0000000000000000FFFFFFFFLLLU};
      x = (x | x << 32) & I{0x3FF00000000FFFF00000000FFFFLLLU};
      x = (x | x << 16) & I{0x30000FF0000FF0000FF0000FF0000FFLLLU};
      x = (x | x << 8) & I{0x300F00F00F00F00F00F00F00F00F00FLLLU};
      x = (x | x << 4) & I{0x30C30C30C30C30C30C30C30C30C30C3LLLU};
      x = (x | x << 2) & I{0x9249249249249249249249249249249LLLU};
      return x;
    }*/
    else
      ARIA_STATIC_ASSERT_FALSE("Type of the coord elements to be encoded is currently not supported");
  };

  return shift(coord.x()) | (shift(coord.y()) << 1) | (shift(coord.z()) << 2);
}

template <std::integral I>
[[nodiscard]] ARIA_HOST_DEVICE inline constexpr Vec3<I> MortonCode<3>::Decode(const I &code) {
  auto unshift = [](I x) {
    if constexpr (sizeof(I) == 4) {
      x &= I{0x9249249U};
      x = (x ^ (x >> 2)) & I{0x30C30C3U};
      x = (x ^ (x >> 4)) & I{0x300F00FU};
      x = (x ^ (x >> 8)) & I{0x30000FFU};
      x = (x ^ (x >> 16)) & I{0x3FFU};
      return x;
    } else if constexpr (sizeof(I) == 8) {
      x &= I{0x1249249249249249LLU};
      x = (x ^ (x >> 2)) & I{0x10C30C30C30C30C3LLU};
      x = (x ^ (x >> 4)) & I{0x100F00F00F00F00FLLU};
      x = (x ^ (x >> 8)) & I{0x1F0000FF0000FFLLU};
      x = (x ^ (x >> 16)) & I{0x1F00000000FFFFLLU};
      x = (x ^ (x >> 32)) & I{0x1FFFFFLLU};
      return x;
    } /*else if constexpr (sizeof(I) == 16) {
      x &= I{0x9249249249249249249249249249249LLLU};
      x = (x ^ (x >> 2)) & I{0x30C30C30C30C30C30C30C30C30C30C3LLLU};
      x = (x ^ (x >> 4)) & I{0x300F00F00F00F00F00F00F00F00F00FLLLU};
      x = (x ^ (x >> 8)) & I{0x30000FF0000FF0000FF0000FF0000FFLLLU};
      x = (x ^ (x >> 16)) & I{0x3FF00000000FFFF00000000FFFFLLLU};
      x = (x ^ (x >> 32)) & I{0x3FF0000000000000000FFFFFFFFLLLU};
      x = (x ^ (x >> 64)) & I{0X3FFFFFFFFFFLLLU};
      return x;
    }*/
  };

  return {unshift(code), unshift(code >> 1), unshift(code >> 2)};
}

} // namespace ARIA
