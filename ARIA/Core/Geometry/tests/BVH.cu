#include "ARIA/BVH.h"

#include <cuda/std/tuple>
#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct Primitives {
  thrust::device_ptr<Vec3f> vertices_;
  thrust::device_ptr<Vec3u> triangles_;
  uint size_;

  ARIA_HOST_DEVICE cuda::std::tuple<Vec3f, Vec3f, Vec3f> operator()(uint i) const {
    Vec3u triangle = triangles_[i];
    Vec3f v0 = vertices_[triangle.x()];
    Vec3f v1 = vertices_[triangle.y()];
    Vec3f v2 = vertices_[triangle.z()];
    return {v0, v1, v2};
  }

  ARIA_HOST_DEVICE uint size() const { return size_; }
};

void TestBVH() {
  // clang-format off
  thrust::device_vector<Vec3f> vertices = {
      // forward
      Vec3f(-0.5, -0.5, -0.5),
      Vec3f( 0.5, -0.5, -0.5),
      Vec3f( 0.5,  0.5, -0.5),
      Vec3f( 0.5,  0.5, -0.5),
      Vec3f(-0.5,  0.5, -0.5),
      Vec3f(-0.5, -0.5, -0.5),
      // back
      Vec3f(-0.5, -0.5,  0.5),
      Vec3f( 0.5, -0.5,  0.5),
      Vec3f( 0.5,  0.5,  0.5),
      Vec3f( 0.5,  0.5,  0.5),
      Vec3f(-0.5,  0.5,  0.5),
      Vec3f(-0.5, -0.5,  0.5),
      // left
      Vec3f(-0.5,  0.5,  0.5),
      Vec3f(-0.5,  0.5, -0.5),
      Vec3f(-0.5, -0.5, -0.5),
      Vec3f(-0.5, -0.5, -0.5),
      Vec3f(-0.5, -0.5,  0.5),
      Vec3f(-0.5,  0.5,  0.5),
      // right
      Vec3f( 0.5,  0.5,  0.5),
      Vec3f( 0.5,  0.5, -0.5),
      Vec3f( 0.5, -0.5, -0.5),
      Vec3f( 0.5, -0.5, -0.5),
      Vec3f( 0.5, -0.5,  0.5),
      Vec3f( 0.5,  0.5,  0.5),
      // down
      Vec3f(-0.5, -0.5, -0.5),
      Vec3f( 0.5, -0.5, -0.5),
      Vec3f( 0.5, -0.5,  0.5),
      Vec3f( 0.5, -0.5,  0.5),
      Vec3f(-0.5, -0.5,  0.5),
      Vec3f(-0.5, -0.5, -0.5),
      // up
      Vec3f(-0.5,  0.5, -0.5),
      Vec3f( 0.5,  0.5, -0.5),
      Vec3f( 0.5,  0.5,  0.5),
      Vec3f( 0.5,  0.5,  0.5),
      Vec3f(-0.5,  0.5,  0.5),
      Vec3f(-0.5,  0.5, -0.5),
  };
  // clang-format on

  // clang-format off
  thrust::device_vector<Vec3u> triangles = {
      Vec3u( 0,  1,  2), Vec3u( 3,  4,  5),
      Vec3u( 6,  7,  8), Vec3u( 9, 10, 11),
      Vec3u(12, 13, 14), Vec3u(15, 16, 17),
      Vec3u(18, 19, 20), Vec3u(21, 22, 23),
      Vec3u(24, 25, 26), Vec3u(27, 28, 29),
      Vec3u(30, 31, 32), Vec3u(33, 34, 35),
  };
  // clang-format on

  Primitives primitives{
      .vertices_ = vertices.data(), .triangles_ = triangles.data(), .size_ = static_cast<uint>(triangles.size())};

  auto fPrimitiveToPos = [] ARIA_HOST_DEVICE(const cuda::std::tuple<Vec3f, Vec3f, Vec3f> &triangle) -> Vec3f {
    const Vec3f &v0 = cuda::std::get<0>(triangle);
    const Vec3f &v1 = cuda::std::get<1>(triangle);
    const Vec3f &v2 = cuda::std::get<2>(triangle);
    return (v0 + v1 + v2) / 3;
  };

  auto fPrimitiveToAABB = [] ARIA_HOST_DEVICE(const cuda::std::tuple<Vec3f, Vec3f, Vec3f> &triangle) -> AABB3f {
    const Vec3f &v0 = cuda::std::get<0>(triangle);
    const Vec3f &v1 = cuda::std::get<1>(triangle);
    const Vec3f &v2 = cuda::std::get<2>(triangle);
    return AABB3f{v0, v1, v2};
  };

  make_bvh_device(primitives, fPrimitiveToPos, fPrimitiveToAABB);
}

} // namespace

TEST(BVH, Base) {
  TestBVH();
}

} // namespace ARIA
