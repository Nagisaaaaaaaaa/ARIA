#include "ARIA/Launcher.h"
#include "ARIA/TensorVector.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct PatternInts {
  int v[2];
};

} // namespace

template <>
struct Mosaic<int, PatternInts> {
  ARIA_HOST_DEVICE PatternInts operator()(const int &v) const { return {.v = {v / 2, v % 2}}; }

  ARIA_HOST_DEVICE int operator()(const PatternInts &v) const { return v.v[0] * 2 + v.v[1]; }
};

namespace {

template <typename TPtr>
ARIA_KERNEL void TestKernel(int size, TPtr ptr) {
  int i = static_cast<int>(threadIdx.x) + static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
  if (i >= size)
    return;

  ptr[i] = i;
}

template <typename TStorage>
void GTestLaunchBase_Kernel() {
  {
    int size = 10;
    TStorage v(size);
    Launcher(TestKernel<std::decay_t<decltype(v.data())>>).overallSize(size).blockSize(256).Launch(size, v.data());
    cuda::device::current::get().synchronize();
    for (int i = 0; i < size; ++i)
      EXPECT_TRUE(v[i] == i);
  }
}

template <typename TStorage>
void GTestLaunchBase_Integral() {
  {
    int size = 10;
    TStorage v(size);
    Launcher(size, [v = v.data()] ARIA_DEVICE(const int &i) { v[i] = i; }).blockSize(256).Launch();
    cuda::device::current::get().synchronize();
    for (int i = 0; i < size; ++i)
      EXPECT_TRUE(v[i] == i);
  }

  {
    int size = 10;
    TStorage v(size);
    Launcher l = Launcher(size, [v = v.data()] ARIA_DEVICE(const int &i) { v[i] = i; }).blockSize(256);
    l.Launch();
    cuda::device::current::get().synchronize();
    for (int i = 0; i < size; ++i)
      EXPECT_TRUE(v[i] == i);
  }

  {
    int size = 10;
    TStorage v(size);
    Launcher l = Launcher(size, [v = v.data()] ARIA_DEVICE(const int &i) { v[i] = i; });
    l.blockSize(256).Launch();
    cuda::device::current::get().synchronize();
    for (int i = 0; i < size; ++i)
      EXPECT_TRUE(v[i] == i);
  }
}

template <typename TStorage>
void GTestLaunchBase_Layout() {
  {
    auto layout = make_layout_major(5, 6);

    TStorage aD(layout);
    TStorage bD(layout);

    for (int i = 0; i < aD.size<0>(); ++i) {
      for (int k = 0; k < aD.size<1>(); ++k) {
        aD(i, k) = 2 * i + 2 * k;
        bD(i, k) = i - k;
      }
    }

    TStorage cD(layout);

    Launcher(aD.layout(), [a = aD.tensor(), b = bD.tensor(),
                           c = cD.tensor()] ARIA_DEVICE(const int &x, const int &y) { c(x, y) = a(x, y) + b(x, y); })
        .blockSize(128)
        .Launch();

    cuda::device::current::get().synchronize();

    for (int i = 0; i < cD.size<0>(); ++i) {
      for (int k = 0; k < cD.size<1>(); ++k) {
        EXPECT_TRUE(cD(i, k) == 3 * i + k);
      }
    }
  }
}

} // namespace

TEST(Launch, Base) {
  using T = int;
  using TMosaic = Mosaic<T, PatternInts>;

  // Kernel.
  {
    GTestLaunchBase_Kernel<thrust::device_vector<T>>();
    GTestLaunchBase_Kernel<VectorDevice<T>>();
    GTestLaunchBase_Kernel<VectorDevice<TMosaic>>();
  }

  // Integral.
  {
    GTestLaunchBase_Integral<thrust::device_vector<T>>();
    GTestLaunchBase_Integral<VectorDevice<T>>();
    GTestLaunchBase_Integral<VectorDevice<TMosaic>>();
  }

  // Layout.
  {
    GTestLaunchBase_Layout<TensorVectorDevice<T, C<2>>>();
    GTestLaunchBase_Layout<TensorVectorDevice<TMosaic, C<2>>>();
  }
}

} // namespace ARIA
