#include "ARIA/Vec.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

class Test {
public:
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, position);
};

} // namespace

TEST(Vec, Base) {
  {
    static_assert(sizeof(Vec1i) == 1 * sizeof(int));
    static_assert(sizeof(Vec1u) == 1 * sizeof(uint));
    static_assert(sizeof(Vec1f) == 1 * sizeof(float));
    static_assert(sizeof(Vec1d) == 1 * sizeof(double));
    static_assert(sizeof(Vec1r) == 1 * sizeof(Real));

    static_assert(sizeof(Vec2i) == 2 * sizeof(int));
    static_assert(sizeof(Vec2u) == 2 * sizeof(uint));
    static_assert(sizeof(Vec2f) == 2 * sizeof(float));
    static_assert(sizeof(Vec2d) == 2 * sizeof(double));
    static_assert(sizeof(Vec2r) == 2 * sizeof(Real));

    static_assert(sizeof(Vec3i) == 3 * sizeof(int));
    static_assert(sizeof(Vec3u) == 3 * sizeof(uint));
    static_assert(sizeof(Vec3f) == 3 * sizeof(float));
    static_assert(sizeof(Vec3d) == 3 * sizeof(double));
    static_assert(sizeof(Vec3r) == 3 * sizeof(Real));

    static_assert(sizeof(Vec4i) == 4 * sizeof(int));
    static_assert(sizeof(Vec4u) == 4 * sizeof(uint));
    static_assert(sizeof(Vec4f) == 4 * sizeof(float));
    static_assert(sizeof(Vec4d) == 4 * sizeof(double));
    static_assert(sizeof(Vec4r) == 4 * sizeof(Real));
  }

  {
    static_assert(vec::detail::is_vec_v<Vec2i>);
    static_assert(vec::detail::is_vec_s_v<Vec2i, 2>);
    static_assert(!vec::detail::is_vec_s_v<Vec2i, 1>);
    static_assert(!vec::detail::is_vec_s_v<Vec2i, 3>);
  }
}

} // namespace ARIA
