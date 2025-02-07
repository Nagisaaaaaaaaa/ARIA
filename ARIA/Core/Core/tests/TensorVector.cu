#include "ARIA/TensorVector.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

using cute::_0;
using cute::_1;
using cute::_2;
using cute::_3;
using cute::_4;

struct PatternFloats {
  float v[2];
};

} // namespace

template <>
struct Mosaic<float, PatternFloats> {
  PatternFloats operator()(const float &v) const { return {.v = {v * (2.0F / 5.0F), v * (3.0F / 5.0F)}}; }

  float operator()(const PatternFloats &v) const { return v.v[0] + v.v[1]; }
};

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

TEST(TensorVector, RankTypeDeduction) {
  // 1D.
  {
    TensorVector<float, SpaceHost> tvH;
    TensorVector<float, SpaceDevice> tvD;
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, _1, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, _1, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, Int<1>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, Int<1>, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, UInt<1>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, UInt<1>, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, std::integral_constant<int, 1>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, std::integral_constant<int, 1>, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, std::integral_constant<uint, 1>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, std::integral_constant<uint, 1>, SpaceDevice>>);
  }

  // 2D.
  {
    TensorVector<float, _2, SpaceHost> tvH;
    TensorVector<float, _2, SpaceDevice> tvD;
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, Int<2>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, Int<2>, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, UInt<2>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, UInt<2>, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, std::integral_constant<int, 2>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, std::integral_constant<int, 2>, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, std::integral_constant<uint, 2>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, std::integral_constant<uint, 2>, SpaceDevice>>);
  }

  // 3D.
  {
    TensorVector<float, _3, SpaceHost> tvH;
    TensorVector<float, _3, SpaceDevice> tvD;
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, Int<3>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, Int<3>, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, UInt<3>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, UInt<3>, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, std::integral_constant<int, 3>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, std::integral_constant<int, 3>, SpaceDevice>>);
    static_assert(std::is_same_v<decltype(tvH), TensorVector<float, std::integral_constant<uint, 3>, SpaceHost>>);
    static_assert(std::is_same_v<decltype(tvD), TensorVector<float, std::integral_constant<uint, 3>, SpaceDevice>>);
  }
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

TEST(TensorVector, AssignmentAndCopyStatic) {
  using T = float;
  using TMosaic = Mosaic<T, PatternFloats>;
  using TLayout = decltype(make_layout_major(C<10>{}));

  // 1D.
  ForEach<MakeTypeArray<                                 //
      Tup<TensorVectorHost<T>, TensorVectorHost<T>>,     //
      Tup<TensorVectorDevice<T>, TensorVectorHost<T>>,   //
      Tup<TensorVectorHost<T>, TensorVectorDevice<T>>,   //
      Tup<TensorVectorDevice<T>, TensorVectorDevice<T>>, //
      //
      Tup<TensorVectorHost<TMosaic>, TensorVectorHost<TMosaic>>,    //
      Tup<TensorVectorDevice<TMosaic>, TensorVectorHost<TMosaic>>,  //
      Tup<TensorVectorHost<TMosaic>, TensorVectorDevice<TMosaic>>,  //
      Tup<TensorVectorDevice<TMosaic>, TensorVectorDevice<TMosaic>> //
      >>([]<typename TVectors>() {
    using TVector0 = tup_elem_t<0, TVectors>;
    using TVector1 = tup_elem_t<1, TVectors>;
    static_assert(std::is_same_v<typename TVector0::value_type, float>);
    static_assert(std::is_same_v<typename TVector1::value_type, float>);

    TVector0 dst;
    TVector1 src;

    for (int i = 0; i < 10; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 10; ++i)
      EXPECT_FLOAT_EQ(dst(i), i);
  });

  // 2D.
  ForEach<MakeTypeArray<                                         //
      Tup<TensorVectorHost<T, _2>, TensorVectorHost<T, _2>>,     //
      Tup<TensorVectorDevice<T, _2>, TensorVectorHost<T, _2>>,   //
      Tup<TensorVectorHost<T, _2>, TensorVectorDevice<T, _2>>,   //
      Tup<TensorVectorDevice<T, _2>, TensorVectorDevice<T, _2>>, //
      //
      Tup<TensorVectorHost<TMosaic, _2>, TensorVectorHost<TMosaic, _2>>,    //
      Tup<TensorVectorDevice<TMosaic, _2>, TensorVectorHost<TMosaic, _2>>,  //
      Tup<TensorVectorHost<TMosaic, _2>, TensorVectorDevice<TMosaic, _2>>,  //
      Tup<TensorVectorDevice<TMosaic, _2>, TensorVectorDevice<TMosaic, _2>> //
      >>([]<typename TVectors>() {
    using TVector0 = tup_elem_t<0, TVectors>;
    using TVector1 = tup_elem_t<1, TVectors>;
    static_assert(std::is_same_v<typename TVector0::value_type, float>);
    static_assert(std::is_same_v<typename TVector1::value_type, float>);

    TVector0 dst;
    TVector1 src;

    for (int i = 0; i < 30; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 30; ++i)
      EXPECT_FLOAT_EQ(dst(i), i);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        src(x, y) = x + 3 * y + 1;

    copy(dst, src);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        EXPECT_FLOAT_EQ(dst(x, y), x + 3 * y + 1);
  });

  // 3D.
  ForEach<MakeTypeArray<                                         //
      Tup<TensorVectorHost<T, _3>, TensorVectorHost<T, _3>>,     //
      Tup<TensorVectorDevice<T, _3>, TensorVectorHost<T, _3>>,   //
      Tup<TensorVectorHost<T, _3>, TensorVectorDevice<T, _3>>,   //
      Tup<TensorVectorDevice<T, _3>, TensorVectorDevice<T, _3>>, //
      //
      Tup<TensorVectorHost<TMosaic, _3>, TensorVectorHost<TMosaic, _3>>,    //
      Tup<TensorVectorDevice<TMosaic, _3>, TensorVectorHost<TMosaic, _3>>,  //
      Tup<TensorVectorHost<TMosaic, _3>, TensorVectorDevice<TMosaic, _3>>,  //
      Tup<TensorVectorDevice<TMosaic, _3>, TensorVectorDevice<TMosaic, _3>> //
      >>([]<typename TVectors>() {
    using TVector0 = tup_elem_t<0, TVectors>;
    using TVector1 = tup_elem_t<1, TVectors>;
    static_assert(std::is_same_v<typename TVector0::value_type, float>);
    static_assert(std::is_same_v<typename TVector1::value_type, float>);

    TVector0 dst;
    TVector1 src;

    for (int i = 0; i < 24; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 24; ++i)
      EXPECT_FLOAT_EQ(dst(i), i);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          src(x, y, z) = x + 2 * y + 3 * z + 1;

    copy(dst, src);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          EXPECT_FLOAT_EQ(dst(x, y, z), x + 2 * y + 3 * z + 1);
  });
}

TEST(TensorVector, AssignmentAndCopyDynamic) {
  using T = float;
  using TMosaic = Mosaic<T, PatternFloats>;

  // 1D.
  ForEach<MakeTypeArray<                                 //
      Tup<TensorVectorHost<T>, TensorVectorHost<T>>,     //
      Tup<TensorVectorDevice<T>, TensorVectorHost<T>>,   //
      Tup<TensorVectorHost<T>, TensorVectorDevice<T>>,   //
      Tup<TensorVectorDevice<T>, TensorVectorDevice<T>>, //
      //
      Tup<TensorVectorHost<TMosaic>, TensorVectorHost<TMosaic>>,    //
      Tup<TensorVectorDevice<TMosaic>, TensorVectorHost<TMosaic>>,  //
      Tup<TensorVectorHost<TMosaic>, TensorVectorDevice<TMosaic>>,  //
      Tup<TensorVectorDevice<TMosaic>, TensorVectorDevice<TMosaic>> //
      >>([]<typename TVectors>() {
    using TVector0 = tup_elem_t<0, TVectors>;
    using TVector1 = tup_elem_t<1, TVectors>;
    static_assert(std::is_same_v<typename TVector0::value_type, float>);
    static_assert(std::is_same_v<typename TVector1::value_type, float>);

    TVector0 dst;
    TVector1 src;
    dst.Realloc(make_layout_major(10));
    src.Realloc(make_layout_major(10));

    for (int i = 0; i < 10; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 10; ++i)
      EXPECT_FLOAT_EQ(dst(i), i);
  });

  // 2D.
  ForEach<MakeTypeArray<                                         //
      Tup<TensorVectorHost<T, _2>, TensorVectorHost<T, _2>>,     //
      Tup<TensorVectorDevice<T, _2>, TensorVectorHost<T, _2>>,   //
      Tup<TensorVectorHost<T, _2>, TensorVectorDevice<T, _2>>,   //
      Tup<TensorVectorDevice<T, _2>, TensorVectorDevice<T, _2>>, //
      //
      Tup<TensorVectorHost<TMosaic, _2>, TensorVectorHost<TMosaic, _2>>,    //
      Tup<TensorVectorDevice<TMosaic, _2>, TensorVectorHost<TMosaic, _2>>,  //
      Tup<TensorVectorHost<TMosaic, _2>, TensorVectorDevice<TMosaic, _2>>,  //
      Tup<TensorVectorDevice<TMosaic, _2>, TensorVectorDevice<TMosaic, _2>> //
      >>([]<typename TVectors>() {
    using TVector0 = tup_elem_t<0, TVectors>;
    using TVector1 = tup_elem_t<1, TVectors>;
    static_assert(std::is_same_v<typename TVector0::value_type, float>);
    static_assert(std::is_same_v<typename TVector1::value_type, float>);

    TVector0 dst;
    TVector1 src;
    dst.Realloc(make_layout_major(5, 6));
    src.Realloc(make_layout_major(5, 6));

    for (int i = 0; i < 30; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 30; ++i)
      EXPECT_FLOAT_EQ(dst(i), i);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        src(x, y) = x + 3 * y + 1;

    copy(dst, src);

    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 5; ++x)
        EXPECT_FLOAT_EQ(dst(x, y), x + 3 * y + 1);
  });

  // 3D.
  ForEach<MakeTypeArray<                                         //
      Tup<TensorVectorHost<T, _3>, TensorVectorHost<T, _3>>,     //
      Tup<TensorVectorDevice<T, _3>, TensorVectorHost<T, _3>>,   //
      Tup<TensorVectorHost<T, _3>, TensorVectorDevice<T, _3>>,   //
      Tup<TensorVectorDevice<T, _3>, TensorVectorDevice<T, _3>>, //
      //
      Tup<TensorVectorHost<TMosaic, _3>, TensorVectorHost<TMosaic, _3>>,    //
      Tup<TensorVectorDevice<TMosaic, _3>, TensorVectorHost<TMosaic, _3>>,  //
      Tup<TensorVectorHost<TMosaic, _3>, TensorVectorDevice<TMosaic, _3>>,  //
      Tup<TensorVectorDevice<TMosaic, _3>, TensorVectorDevice<TMosaic, _3>> //
      >>([]<typename TVectors>() {
    using TVector0 = tup_elem_t<0, TVectors>;
    using TVector1 = tup_elem_t<1, TVectors>;
    static_assert(std::is_same_v<typename TVector0::value_type, float>);
    static_assert(std::is_same_v<typename TVector1::value_type, float>);

    TVector0 dst;
    TVector1 src;
    dst.Realloc(make_layout_major(2, 3, 4));
    src.Realloc(make_layout_major(2, 3, 4));

    for (int i = 0; i < 24; ++i)
      src(i) = i;

    copy(dst, src);

    for (int i = 0; i < 24; ++i)
      EXPECT_FLOAT_EQ(dst(i), i);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          src(x, y, z) = x + 2 * y + 3 * z + 1;

    copy(dst, src);

    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 2; ++x)
          EXPECT_FLOAT_EQ(dst(x, y, z), x + 2 * y + 3 * z + 1);
  });
}

} // namespace ARIA
