#include "ARIA/TensorVector.h"

#include <gtest/gtest.h>

#include <queue>

namespace ARIA {

namespace {

using cute::_0;
using cute::_1;
using cute::_2;
using cute::_3;
using cute::_4;

std::queue<uint> q;

void CheckQ() {
  // clang-format off
  EXPECT_EQ(q.front(), 0); q.pop(); EXPECT_EQ(q.front(), 0); q.pop();
  EXPECT_EQ(q.front(), 1); q.pop(); EXPECT_EQ(q.front(), 0); q.pop();
  EXPECT_EQ(q.front(), 0); q.pop(); EXPECT_EQ(q.front(), 1); q.pop();
  EXPECT_EQ(q.front(), 1); q.pop(); EXPECT_EQ(q.front(), 1); q.pop();
  EXPECT_EQ(q.front(), 0); q.pop(); EXPECT_EQ(q.front(), 2); q.pop();
  EXPECT_EQ(q.front(), 1); q.pop(); EXPECT_EQ(q.front(), 2); q.pop();
  EXPECT_EQ(q.front(), 0); q.pop(); EXPECT_EQ(q.front(), 3); q.pop();
  EXPECT_EQ(q.front(), 1); q.pop(); EXPECT_EQ(q.front(), 3); q.pop();
  // clang-format on
};

} // namespace

TEST(TensorVector, HostStatic) {
  auto layout = make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})));
  auto tv = make_tensor_vector<float, SpaceHost>(layout);

  static_assert(!is_tensor_vector_v<decltype(layout)>);
  static_assert(is_tensor_vector_v<decltype(tv)>);

  static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
  static_assert(tv.rank == 2);
  static_assert(is_layout_const_size_v<decltype(tv)::Layout>);
  static_assert(is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

  static_assert(tv.size() == 8);
  static_assert(tv.size<0>() == 2);
  static_assert(tv.size<1>() == 4);

  static_assert(std::is_same_v<decltype(tv.layout()), decltype(layout)>);
  EXPECT_TRUE(tv.layout() == layout);

  auto tensor = tv.tensor();
  static_assert(std::is_same_v<decltype(tensor.layout()), decltype(layout)>);
  EXPECT_TRUE(tensor.layout() == layout);

  auto rawTensor = tv.rawTensor();
  static_assert(std::is_same_v<decltype(rawTensor.layout()), decltype(layout)>);
  EXPECT_TRUE(rawTensor.layout() == layout);
  static_assert(std::is_same_v<decltype(rawTensor), decltype(tensor)>);
}

TEST(TensorVector, HostDynamic) {
  auto layout = make_layout(make_shape(2, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})));
  auto tv = make_tensor_vector<float, SpaceHost>(layout);

  static_assert(!is_tensor_vector_v<decltype(layout)>);
  static_assert(is_tensor_vector_v<decltype(tv)>);

  static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
  static_assert(tv.rank == 2);
  static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
  static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

  EXPECT_TRUE(tv.size() == 8);
  EXPECT_TRUE(tv.size<0>() == 2);
  static_assert(tv.size<1>() == 4);

  EXPECT_TRUE(tv.layout() == layout);

  auto tensor = tv.tensor();
  EXPECT_TRUE(tensor.layout() == layout);

  auto rawTensor = tv.rawTensor();
  EXPECT_TRUE(rawTensor.layout() == layout);
  static_assert(std::is_same_v<decltype(rawTensor), decltype(tensor)>);
}

TEST(TensorVector, DeviceStatic) {
  auto layout = make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})));
  auto tv = make_tensor_vector<float, SpaceDevice>(layout);

  static_assert(!is_tensor_vector_v<decltype(layout)>);
  static_assert(is_tensor_vector_v<decltype(tv)>);

  static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
  static_assert(tv.rank == 2);
  static_assert(is_layout_const_size_v<decltype(tv)::Layout>);
  static_assert(is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

  static_assert(tv.size() == 8);
  static_assert(tv.size<0>() == 2);
  static_assert(tv.size<1>() == 4);

  static_assert(std::is_same_v<decltype(tv.layout()), decltype(layout)>);
  EXPECT_TRUE(tv.layout() == layout);

  auto tensor = tv.tensor();
  static_assert(std::is_same_v<decltype(tensor.layout()), decltype(layout)>);
  EXPECT_TRUE(tensor.layout() == layout);

  auto rawTensor = tv.rawTensor();
  static_assert(std::is_same_v<decltype(rawTensor.layout()), decltype(layout)>);
  EXPECT_TRUE(rawTensor.layout() == layout);
  static_assert(std::is_same_v<decltype(rawTensor), decltype(cute::make_tensor(tensor.data().get(), layout))>);
}

TEST(TensorVector, DeviceDynamic) {
  auto layout = make_layout(make_shape(2, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})));
  auto tv = make_tensor_vector<float, SpaceDevice>(layout);

  static_assert(!is_tensor_vector_v<decltype(layout)>);
  static_assert(is_tensor_vector_v<decltype(tv)>);

  static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
  static_assert(tv.rank == 2);
  static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
  static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

  EXPECT_TRUE(tv.size() == 8);
  EXPECT_TRUE(tv.size<0>() == 2);
  static_assert(tv.size<1>() == 4);

  EXPECT_TRUE(tv.layout() == layout);

  auto tensor = tv.tensor();
  EXPECT_TRUE(tensor.layout() == layout);

  auto rawTensor = tv.rawTensor();
  EXPECT_TRUE(rawTensor.layout() == layout);
  static_assert(std::is_same_v<decltype(rawTensor), decltype(cute::make_tensor(tensor.data().get(), layout))>);
}

TEST(TensorVector, Major) {
  // 1D.
  {
    TensorVector<float, SpaceHost> tv;

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
    static_assert(tv.rank == 1);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    tv.Realloc(make_layout_major(10));
    EXPECT_TRUE(tv.size() == 10);
    EXPECT_TRUE(tv.size<0>() == 10);
  }

  {
    TensorVector<float, SpaceDevice> tv;

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
    static_assert(tv.rank == 1);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    tv.Realloc(make_layout_major(10));
    EXPECT_TRUE(tv.size() == 10);
    EXPECT_TRUE(tv.size<0>() == 10);
  }

  {
    auto tv = make_tensor_vector<float, SpaceHost>(make_layout_major(0));

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
    static_assert(tv.rank == 1);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    tv.Realloc(make_layout_major(10));
    EXPECT_TRUE(tv.size() == 10);
    EXPECT_TRUE(tv.size<0>() == 10);
  }

  {
    auto tv = make_tensor_vector<float, SpaceDevice>(make_layout_major(0));

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
    static_assert(tv.rank == 1);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    tv.Realloc(make_layout_major(10));
    EXPECT_TRUE(tv.size() == 10);
    EXPECT_TRUE(tv.size<0>() == 10);
  }

  // 2D.
  {
    TensorVector<float, _2, SpaceHost> tv;

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
    static_assert(tv.rank == 2);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    tv.Realloc(make_layout_major(4, 5));
    EXPECT_TRUE(tv.size() == 20);
    EXPECT_TRUE(tv.size<0>() == 4);
    EXPECT_TRUE(tv.size<1>() == 5);
  }

  {
    TensorVector<float, _2, SpaceDevice> tv;

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
    static_assert(tv.rank == 2);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    tv.Realloc(make_layout_major(4, 5));
    EXPECT_TRUE(tv.size() == 20);
    EXPECT_TRUE(tv.size<0>() == 4);
    EXPECT_TRUE(tv.size<1>() == 5);
  }

  {
    auto tv = make_tensor_vector<float, SpaceHost>(make_layout_major(0, 0));

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
    static_assert(tv.rank == 2);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    tv.Realloc(make_layout_major(4, 5));
    EXPECT_TRUE(tv.size() == 20);
    EXPECT_TRUE(tv.size<0>() == 4);
    EXPECT_TRUE(tv.size<1>() == 5);
  }

  {
    auto tv = make_tensor_vector<float, SpaceDevice>(make_layout_major(0, 0));

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
    static_assert(tv.rank == 2);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    tv.Realloc(make_layout_major(4, 5));
    EXPECT_TRUE(tv.size() == 20);
    EXPECT_TRUE(tv.size<0>() == 4);
    EXPECT_TRUE(tv.size<1>() == 5);
  }

  // 3D.
  {
    TensorVector<float, _3, SpaceHost> tv;

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
    static_assert(tv.rank == 3);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    EXPECT_TRUE(tv.size<2>() == 0);
    tv.Realloc(make_layout_major(2, 3, 4));
    EXPECT_TRUE(tv.size() == 24);
    EXPECT_TRUE(tv.size<0>() == 2);
    EXPECT_TRUE(tv.size<1>() == 3);
    EXPECT_TRUE(tv.size<2>() == 4);
  }

  {
    TensorVector<float, _3, SpaceDevice> tv;

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
    static_assert(tv.rank == 3);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    EXPECT_TRUE(tv.size<2>() == 0);
    tv.Realloc(make_layout_major(2, 3, 4));
    EXPECT_TRUE(tv.size() == 24);
    EXPECT_TRUE(tv.size<0>() == 2);
    EXPECT_TRUE(tv.size<1>() == 3);
    EXPECT_TRUE(tv.size<2>() == 4);
  }

  {
    auto tv = make_tensor_vector<float, SpaceHost>(make_layout_major(0, 0, 0));

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
    static_assert(tv.rank == 3);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    EXPECT_TRUE(tv.size<2>() == 0);
    tv.Realloc(make_layout_major(2, 3, 4));
    EXPECT_TRUE(tv.size() == 24);
    EXPECT_TRUE(tv.size<0>() == 2);
    EXPECT_TRUE(tv.size<1>() == 3);
    EXPECT_TRUE(tv.size<2>() == 4);
  }

  {
    auto tv = make_tensor_vector<float, SpaceDevice>(make_layout_major(0, 0, 0));

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
    static_assert(tv.rank == 3);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    EXPECT_TRUE(tv.size<2>() == 0);
    tv.Realloc(make_layout_major(2, 3, 4));
    EXPECT_TRUE(tv.size() == 24);
    EXPECT_TRUE(tv.size<0>() == 2);
    EXPECT_TRUE(tv.size<1>() == 3);
    EXPECT_TRUE(tv.size<2>() == 4);
  }
}

TEST(TensorVector, HostAndDeviceTensorVector) {
  // 1D.
  {
    TensorVectorHost<float> tv;
    static_assert(std::is_same_v<decltype(tv), TensorVector<float, SpaceHost>>);

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
    static_assert(tv.rank == 1);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    tv.Realloc(make_layout_major(10));
    EXPECT_TRUE(tv.size() == 10);
    EXPECT_TRUE(tv.size<0>() == 10);
  }

  {
    TensorVectorDevice<float> tv;
    static_assert(std::is_same_v<decltype(tv), TensorVector<float, SpaceDevice>>);

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
    static_assert(tv.rank == 1);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    tv.Realloc(make_layout_major(10));
    EXPECT_TRUE(tv.size() == 10);
    EXPECT_TRUE(tv.size<0>() == 10);
  }

  // 2D.
  {
    TensorVectorHost<float, _2> tv;
    static_assert(std::is_same_v<decltype(tv), TensorVector<float, _2, SpaceHost>>);

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
    static_assert(tv.rank == 2);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    tv.Realloc(make_layout_major(4, 5));
    EXPECT_TRUE(tv.size() == 20);
    EXPECT_TRUE(tv.size<0>() == 4);
    EXPECT_TRUE(tv.size<1>() == 5);
  }

  {
    TensorVectorDevice<float, _2> tv;
    static_assert(std::is_same_v<decltype(tv), TensorVector<float, _2, SpaceDevice>>);

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
    static_assert(tv.rank == 2);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    tv.Realloc(make_layout_major(4, 5));
    EXPECT_TRUE(tv.size() == 20);
    EXPECT_TRUE(tv.size<0>() == 4);
    EXPECT_TRUE(tv.size<1>() == 5);
  }

  // 3D.
  {
    TensorVectorHost<float, _3> tv;
    static_assert(std::is_same_v<decltype(tv), TensorVector<float, _3, SpaceHost>>);

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceHost>);
    static_assert(tv.rank == 3);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    EXPECT_TRUE(tv.size<2>() == 0);
    tv.Realloc(make_layout_major(2, 3, 4));
    EXPECT_TRUE(tv.size() == 24);
    EXPECT_TRUE(tv.size<0>() == 2);
    EXPECT_TRUE(tv.size<1>() == 3);
    EXPECT_TRUE(tv.size<2>() == 4);
  }

  {
    TensorVectorDevice<float, _3> tv;
    static_assert(std::is_same_v<decltype(tv), TensorVector<float, _3, SpaceDevice>>);

    static_assert(std::is_same_v<decltype(tv)::Space, SpaceDevice>);
    static_assert(tv.rank == 3);
    static_assert(!is_layout_const_size_v<decltype(tv)::Layout>);
    static_assert(!is_layout_const_cosize_safe_v<decltype(tv)::Layout>);

    EXPECT_TRUE(tv.size() == 0);
    EXPECT_TRUE(tv.size<0>() == 0);
    EXPECT_TRUE(tv.size<1>() == 0);
    EXPECT_TRUE(tv.size<2>() == 0);
    tv.Realloc(make_layout_major(2, 3, 4));
    EXPECT_TRUE(tv.size() == 24);
    EXPECT_TRUE(tv.size<0>() == 2);
    EXPECT_TRUE(tv.size<1>() == 3);
    EXPECT_TRUE(tv.size<2>() == 4);
  }
}

TEST(TensorVector, Mirrored) {
  {
    auto layout = make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})));
    auto tv = make_tensor_vector<float, SpaceHost>(layout);
    static_assert(std::is_same_v<decltype(tv), TensorVectorHost<float, decltype(layout)>>);

    using TLayout = decltype(layout);

    static_assert(std::is_same_v<decltype(tv)::Mirrored<SpaceDevice>, TensorVector<float, _2, SpaceDevice, TLayout>>);
    // static_assert(std::is_same_v<decltype(tv)::Mirrored<double>, TensorVector<float, _3, SpaceHost, TLayout>>);
  }

  {
    auto layout = make_layout(make_shape(2, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})));
    auto tv = make_tensor_vector<float, SpaceHost>(layout);
    static_assert(std::is_same_v<decltype(tv), TensorVectorHost<float, decltype(layout)>>);

    using TLayout = decltype(layout);

    static_assert(std::is_same_v<decltype(tv)::Mirrored<SpaceDevice>, TensorVector<float, _2, SpaceDevice, TLayout>>);
    // static_assert(std::is_same_v<decltype(tv)::Mirrored<double>, TensorVector<float, _3, SpaceHost, TLayout>>);
  }

  {
    auto layout = make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})));
    auto tv = make_tensor_vector<float, SpaceDevice>(layout);
    static_assert(std::is_same_v<decltype(tv), TensorVectorDevice<float, decltype(layout)>>);

    using TLayout = decltype(layout);

    static_assert(std::is_same_v<decltype(tv)::Mirrored<SpaceHost>, TensorVector<float, _2, SpaceHost, TLayout>>);
    // static_assert(std::is_same_v<decltype(tv)::Mirrored<double>, TensorVector<float, _3, SpaceDevice, TLayout>>);
  }

  {
    auto layout = make_layout(make_shape(2, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})));
    auto tv = make_tensor_vector<float, SpaceDevice>(layout);
    static_assert(std::is_same_v<decltype(tv), TensorVectorDevice<float, decltype(layout)>>);

    using TLayout = decltype(layout);

    static_assert(std::is_same_v<decltype(tv)::Mirrored<SpaceHost>, TensorVector<float, _2, SpaceHost, TLayout>>);
    // static_assert(std::is_same_v<decltype(tv)::Mirrored<double>, TensorVector<float, _3, SpaceDevice, TLayout>>);
  }
}

TEST(TensorVector, Copy) {
  // 1D.
  { // Host <- host.
    TensorVectorHost<float> dst;
    TensorVectorHost<float> src;
    dst.Realloc(make_layout_major(10));
    src.Realloc(make_layout_major(10));

    for (int i = 0; i < 10; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 10; ++i)
      EXPECT_TRUE(dst(i) == i);
  }

  { // Host <- device.
    TensorVectorHost<float> dst;
    TensorVectorDevice<float> src;
    dst.Realloc(make_layout_major(10));
    src.Realloc(make_layout_major(10));

    for (int i = 0; i < 10; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 10; ++i)
      EXPECT_TRUE(dst(i) == i);
  }

  { // Device <- host.
    TensorVectorDevice<float> dst;
    TensorVectorHost<float> src;
    dst.Realloc(make_layout_major(10));
    src.Realloc(make_layout_major(10));

    for (int i = 0; i < 10; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 10; ++i)
      EXPECT_TRUE(dst(i) == i);
  }

  { // Device <- device.
    TensorVectorDevice<float> dst;
    TensorVectorDevice<float> src;
    dst.Realloc(make_layout_major(10));
    src.Realloc(make_layout_major(10));

    for (int i = 0; i < 10; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 10; ++i)
      EXPECT_TRUE(dst(i) == i);
  }

  // 2D.
  { // Host <- host.
    TensorVectorHost<float, _2> dst;
    TensorVectorHost<float, _2> src;
    dst.Realloc(make_layout_major(5, 6));
    src.Realloc(make_layout_major(5, 6));

    for (int i = 0; i < 30; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 30; ++i)
      EXPECT_TRUE(dst(i) == i);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        src(x, y) = x + 3 * y + 1;

    copy(dst, src);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        EXPECT_TRUE(dst(x, y) == x + 3 * y + 1);
  }

  { // Host <- device.
    TensorVectorHost<float, _2> dst;
    TensorVectorDevice<float, _2> src;
    dst.Realloc(make_layout_major(5, 6));
    src.Realloc(make_layout_major(5, 6));

    for (int i = 0; i < 30; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 30; ++i)
      EXPECT_TRUE(dst(i) == i);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        src(x, y) = x + 3 * y + 1;

    copy(dst, src);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        EXPECT_TRUE(dst(x, y) == x + 3 * y + 1);
  }

  { // Device <- host.
    TensorVectorDevice<float, _2> dst;
    TensorVectorHost<float, _2> src;
    dst.Realloc(make_layout_major(5, 6));
    src.Realloc(make_layout_major(5, 6));

    for (int i = 0; i < 30; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 30; ++i)
      EXPECT_TRUE(dst(i) == i);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        src(x, y) = x + 3 * y + 1;

    copy(dst, src);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        EXPECT_TRUE(dst(x, y) == x + 3 * y + 1);
  }

  { // Device <- device.
    TensorVectorDevice<float, _2> dst;
    TensorVectorDevice<float, _2> src;
    dst.Realloc(make_layout_major(5, 6));
    src.Realloc(make_layout_major(5, 6));

    for (int i = 0; i < 30; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 30; ++i)
      EXPECT_TRUE(dst(i) == i);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        src(x, y) = x + 3 * y + 1;

    copy(dst, src);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        EXPECT_TRUE(dst(x, y) == x + 3 * y + 1);
  }

  // 3D.
  { // Host <- host.
    TensorVectorHost<float, _3> dst;
    TensorVectorHost<float, _3> src;
    dst.Realloc(make_layout_major(2, 3, 4));
    src.Realloc(make_layout_major(2, 3, 4));

    for (int i = 0; i < 24; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 24; ++i)
      EXPECT_TRUE(dst(i) == i);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          src(x, y, z) = x + 2 * y + 3 * z + 1;

    copy(dst, src);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          EXPECT_TRUE(dst(x, y, z) == x + 2 * y + 3 * z + 1);
  }

  { // Host <- device.
    TensorVectorHost<float, _3> dst;
    TensorVectorDevice<float, _3> src;
    dst.Realloc(make_layout_major(2, 3, 4));
    src.Realloc(make_layout_major(2, 3, 4));

    for (int i = 0; i < 24; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 24; ++i)
      EXPECT_TRUE(dst(i) == i);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          src(x, y, z) = x + 2 * y + 3 * z + 1;

    copy(dst, src);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          EXPECT_TRUE(dst(x, y, z) == x + 2 * y + 3 * z + 1);
  }

  { // Device <- host.
    TensorVectorDevice<float, _3> dst;
    TensorVectorHost<float, _3> src;
    dst.Realloc(make_layout_major(2, 3, 4));
    src.Realloc(make_layout_major(2, 3, 4));

    for (int i = 0; i < 24; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 24; ++i)
      EXPECT_TRUE(dst(i) == i);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          src(x, y, z) = x + 2 * y + 3 * z + 1;

    copy(dst, src);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          EXPECT_TRUE(dst(x, y, z) == x + 2 * y + 3 * z + 1);
  }

  { // Device <- device.
    TensorVectorDevice<float, _3> dst;
    TensorVectorDevice<float, _3> src;
    dst.Realloc(make_layout_major(2, 3, 4));
    src.Realloc(make_layout_major(2, 3, 4));

    for (int i = 0; i < 24; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 24; ++i)
      EXPECT_TRUE(dst(i) == i);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          src(x, y, z) = x + 2 * y + 3 * z + 1;

    copy(dst, src);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          EXPECT_TRUE(dst(x, y, z) == x + 2 * y + 3 * z + 1);
  }
}

} // namespace ARIA
