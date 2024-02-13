#include "ARIA/Launcher.h"
#include "ARIA/TensorVector.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

ARIA_KERNEL static void TestKernel(int size, thrust::device_ptr<int> ptr) {
  int i = static_cast<int>(threadIdx.x) + static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
  if (i >= size)
    return;

  ptr[i] = i;
}

void GTestLaunchBase() {
  cuda::device_t device = cuda::device::current::get();

  // Kernel.
  {
    int size = 10;
    thrust::device_vector<int> v(size);
    Launcher(TestKernel).overallSize(size).blockSize(256).Launch(size, v.data());
    device.synchronize();
    for (int i = 0; i < size; ++i)
      EXPECT_TRUE(v[i] == i);
  }

  // Integral.
  {
    int size = 10;
    thrust::device_vector<int> v(size);
    Launcher(size, [v = v.data()] ARIA_DEVICE(const int &i) { v[i] = i; }).blockSize(256).Launch();
    device.synchronize();
    for (int i = 0; i < size; ++i)
      EXPECT_TRUE(v[i] == i);
  }

  {
    int size = 10;
    thrust::device_vector<int> v(size);
    Launcher l = Launcher(size, [v = v.data()] ARIA_DEVICE(const int &i) { v[i] = i; }).blockSize(256);
    l.Launch();
    device.synchronize();
    for (int i = 0; i < size; ++i)
      EXPECT_TRUE(v[i] == i);
  }

  {
    int size = 10;
    thrust::device_vector<int> v(size);
    Launcher l = Launcher(size, [v = v.data()] ARIA_DEVICE(const int &i) { v[i] = i; });
    l.blockSize(256).Launch();
    device.synchronize();
    for (int i = 0; i < size; ++i)
      EXPECT_TRUE(v[i] == i);
  }

  // Layout.
  {
    auto layout = make_layout_major(5, 6);

    auto aD = make_tensor_vector<int, SpaceDevice>(layout);
    auto bD = make_tensor_vector<int, SpaceDevice>(layout);

    for (int i = 0; i < aD.size<0>(); ++i) {
      for (int k = 0; k < aD.size<1>(); ++k) {
        aD(i, k) = 2 * i + 2 * k;
        bD(i, k) = i - k;
      }
    }

    auto cD = make_tensor_vector<int, SpaceDevice>(layout);

    Launcher(aD.layout(), [a = aD.tensor(), b = bD.tensor(),
                           c = cD.tensor()] ARIA_DEVICE(const int &x, const int &y) { c(x, y) = a(x, y) + b(x, y); })
        .blockSize(128)
        .Launch();

    device.synchronize();

    for (int i = 0; i < cD.size<0>(); ++i) {
      for (int k = 0; k < cD.size<1>(); ++k) {
        EXPECT_TRUE(cD(i, k) == 3 * i + k);
      }
    }
  }
}

} // namespace

TEST(Launch, Base) {
  GTestLaunchBase();
}

} // namespace ARIA
