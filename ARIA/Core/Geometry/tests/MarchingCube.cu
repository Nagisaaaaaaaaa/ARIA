#include "ARIA/MarchingCube.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(MarchingCube, Base) {
  // 1D.
  {
    using MC = MarchingCube<1>;
    Vec1r positions[2] = {Vec1r{0.0_R}, Vec1r{2.0_R}};
    Real values[2] = {0.0_R, 0.0_R};
    MC::Extract([&](uint i) { return positions[i]; }, [&](uint i) { return values[i]; }, 0.5_R,
                [](cuda::std::array<Vec1r, 1> primitiveVertices) { fmt::println("{}", primitiveVertices[0].x()); });
  }
}

} // namespace ARIA
