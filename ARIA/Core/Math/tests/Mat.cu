#include "ARIA/Array.h"
#include "ARIA/Mat.h"
#include "ARIA/Vector.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

class Test {
public:
  ARIA_PROP_PREFAB_MAT(public, public, , Mat3r, rotationMat);
};

} // namespace

TEST(Mat, Base) {
  {
    static_assert(sizeof(Mat1i) == 1 * 1 * sizeof(int));
    static_assert(sizeof(Mat1u) == 1 * 1 * sizeof(uint));
    static_assert(sizeof(Mat1f) == 1 * 1 * sizeof(float));
    static_assert(sizeof(Mat1d) == 1 * 1 * sizeof(double));
    static_assert(sizeof(Mat1r) == 1 * 1 * sizeof(Real));

    static_assert(sizeof(Mat2i) == 2 * 2 * sizeof(int));
    static_assert(sizeof(Mat2u) == 2 * 2 * sizeof(uint));
    static_assert(sizeof(Mat2f) == 2 * 2 * sizeof(float));
    static_assert(sizeof(Mat2d) == 2 * 2 * sizeof(double));
    static_assert(sizeof(Mat2r) == 2 * 2 * sizeof(Real));

    static_assert(sizeof(Mat3i) == 3 * 3 * sizeof(int));
    static_assert(sizeof(Mat3u) == 3 * 3 * sizeof(uint));
    static_assert(sizeof(Mat3f) == 3 * 3 * sizeof(float));
    static_assert(sizeof(Mat3d) == 3 * 3 * sizeof(double));
    static_assert(sizeof(Mat3r) == 3 * 3 * sizeof(Real));

    static_assert(sizeof(Mat4i) == 4 * 4 * sizeof(int));
    static_assert(sizeof(Mat4u) == 4 * 4 * sizeof(uint));
    static_assert(sizeof(Mat4f) == 4 * 4 * sizeof(float));
    static_assert(sizeof(Mat4d) == 4 * 4 * sizeof(double));
    static_assert(sizeof(Mat4r) == 4 * 4 * sizeof(Real));
  }

  {
    static_assert(mat::detail::is_mat_v<Mat2i>);
    static_assert(mat::detail::is_mat_rc_v<Mat2i, 2, 2>);
    static_assert(!mat::detail::is_mat_rc_v<Mat2i, 2, 3>);
    static_assert(!mat::detail::is_mat_rc_v<Mat2i, 3, 2>);
    static_assert(!mat::detail::is_mat_rc_v<Mat2i, 3, 3>);
  }
}

TEST(Mat, Mosaic) {
  using namespace mosaic::detail;

  using T = Mat<int, 2, 3>;
  using TMosaic = MatMosaic<int, 2, 3>;
  static_assert(is_mosaic_v<TMosaic> && is_valid_mosaic_v<TMosaic>);

  using TMosaic1 = MatMosaic<T>;
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
      T value;
      value << 0 * i, 1 * i, 2 * i, 3 * i, 4 * i, 5 * i;

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

} // namespace ARIA
