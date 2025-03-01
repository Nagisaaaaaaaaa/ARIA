#include "ARIA/Array.h"
#include "ARIA/Vec.h"
#include "ARIA/Vector.h"

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

TEST(Vec, Cast) {
  // To `Tec`.
  {
    static_assert(std::is_same_v<decltype(ToTec(Vec1u{5})), Tec<uint>>);
    Tec<uint> c = ToTec(Vec1u{5});
    EXPECT_EQ(get<0>(c), 5);
  }

  {
    static_assert(std::is_same_v<decltype(ToTec(Vec2i{5, 6})), Tec<int, int>>);
    Tec<int, int> c = ToTec(Vec2i{5, 6});
    EXPECT_EQ(get<0>(c), 5);
    EXPECT_EQ(get<1>(c), 6);
  }

  {
    static_assert(std::is_same_v<decltype(ToTec(Vec3<size_t>{5, 6, 30})), Tec<size_t, size_t, size_t>>);
    Tec<size_t, size_t, size_t> c = ToTec(Vec3<size_t>{5, 6, 30});
    EXPECT_EQ(get<0>(c), 5);
    EXPECT_EQ(get<1>(c), 6);
    EXPECT_EQ(get<2>(c), 30);
  }

  // To `Vec`.
  {
    static_assert(std::is_same_v<decltype(ToVec(Tec{5U})), Vec1u>);
    static_assert(std::is_same_v<decltype(ToVec(Tec{C<5U>{}})), Vec1u>);
    Vec1u v0 = ToVec(Tec{5U});
    Vec1u v1 = ToVec(Tec{C<5U>{}});
    EXPECT_EQ(v0.x(), 5);
    EXPECT_EQ(v1.x(), 5);
  }

  {
    static_assert(std::is_same_v<decltype(ToVec(Tec{5, 6})), Vec2i>);
    static_assert(std::is_same_v<decltype(ToVec(Tec{5, C<6>{}})), Vec2i>);
    Vec2i v0 = ToVec(Tec{5, 6});
    Vec2i v1 = ToVec(Tec{5, C<6>{}});
    EXPECT_EQ(v0.x(), 5);
    EXPECT_EQ(v0.y(), 6);
    EXPECT_EQ(v1.x(), 5);
    EXPECT_EQ(v1.y(), 6);
  }

  {
    static_assert(std::is_same_v<decltype(ToVec(Tec{size_t{5}, size_t{6}, size_t{30}})), Vec3<size_t>>);
    static_assert(std::is_same_v<decltype(ToVec(Tec{size_t{5}, C<size_t{6}>{}, size_t{30}})), Vec3<size_t>>);
    Vec3<size_t> v0 = ToVec(Tec{size_t{5}, size_t{6}, size_t{30}});
    Vec3<size_t> v1 = ToVec(Tec{size_t{5}, C<size_t{6}>{}, size_t{30}});
    EXPECT_EQ(v0.x(), 5);
    EXPECT_EQ(v0.y(), 6);
    EXPECT_EQ(v0.z(), 30);
    EXPECT_EQ(v1.x(), 5);
    EXPECT_EQ(v1.y(), 6);
    EXPECT_EQ(v1.z(), 30);
  }
}

TEST(Vec, Mosaic) {
  using namespace mosaic::detail;

  using T = Vec<int, 3>;
  using TMosaic = VecMosaic<int, 3>;
  static_assert(is_mosaic_v<TMosaic> && is_valid_mosaic_v<TMosaic>);

  using TMosaic1 = VecMosaic<T>;
  using TMosaic2 = mosaic_t<T>;
  static_assert(std::is_same_v<TMosaic, TMosaic1>);
  static_assert(std::is_same_v<TMosaic, TMosaic2>);

  Array<TMosaic, 5> vec0;
  static_assert(std::is_same_v<decltype(vec0), MosaicArray<TMosaic, 5>>);
  VectorHost<TMosaic> vec1(5);
  static_assert(std::is_same_v<decltype(vec1), MosaicVector<TMosaic, SpaceHost>>);
  VectorDevice<TMosaic> vec2(5);
  static_assert(std::is_same_v<decltype(vec2), MosaicVector<TMosaic, SpaceDevice>>);

  auto testVec = [](auto &vec) {
    for (int i = 0; i < 5; ++i) {
      static_assert(Property<decltype(vec[i])>);
      T value{0 * i, 1 * i, 2 * i};

      vec[i] = value;  // Set.
      T vecI = vec[i]; // Get.
      EXPECT_EQ(vecI, value);
      EXPECT_EQ(vec[i], value);
    }
  };
  testVec(vec0);
  testVec(vec1);
  testVec(vec2);
}

TEST(Vec, Format) {
  {
    std::string s = fmt::format("{}", Vec1u{233U});
    EXPECT_EQ(s, "(233)");
  }

  {
    std::string s = fmt::format("{}", Vec2i{100, -100});
    EXPECT_EQ(s, "(100, -100)");
  }

  {
    std::string s = fmt::format("{}", Vec3f{0.1, -1.2, 2.3});
    EXPECT_EQ(s, "(0.1, -1.2, 2.3)");
  }
}

} // namespace ARIA
