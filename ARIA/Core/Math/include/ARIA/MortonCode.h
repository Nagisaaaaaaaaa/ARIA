#pragma once

#include "ARIA/Vec.h"

/// \details The Morton code implementations are based on:
/// 1. http://www.graphics.stanford.edu/~seander/bithacks.html,
/// 2. https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits,
/// 3. https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints.

#if 0
// 2D 32

static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
static const unsigned int S[] = {1, 2, 4, 8};

unsigned int x; // Interleave lower 16 bits of x and y, so the bits of x
unsigned int y; // are in the even positions and bits from y in the odd;
unsigned int z; // z gets the resulting 32-bit Morton Number.
                // x and y must initially be less than 65536.

x = (x | (x << S[3])) & B[3];
x = (x | (x << S[2])) & B[2];
x = (x | (x << S[1])) & B[1];
x = (x | (x << S[0])) & B[0];

y = (y | (y << S[3])) & B[3];
y = (y | (y << S[2])) & B[2];
y = (y | (y << S[1])) & B[1];
y = (y | (y << S[0])) & B[0];

z = x | (y << 1);



// 2D 64

void xy2d_morton(uint64_t x, uint64_t y, uint64_t *d)
{
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;

    y = (y | (y << 16)) & 0x0000FFFF0000FFFF;
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F;
    y = (y | (y << 2)) & 0x3333333333333333;
    y = (y | (y << 1)) & 0x5555555555555555;

    *d = x | (y << 1);
}



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
[[nodiscard]] ARIA_HOST_DEVICE I MortonCode(const Vec2<I> &c) {}

} // namespace ARIA
