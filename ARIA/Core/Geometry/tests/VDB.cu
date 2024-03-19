#include "ARIA/Launcher.h"
#include "ARIA/VDB.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

void TestVDBHandle() {
  size_t size = 1LLU * 1024LLU * 1024LLU * 1024LLU; // 1GB
  EXPECT_EQ(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size), cudaSuccess);

  using Handle = VDBHandle<float, 2, SpaceDevice>;
  Handle handle = Handle::Create();

  const size_t n = 10000;

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { handle.value({i, 0}) = 233; }).Launch();

  cuda::device::current::get().synchronize();

  handle.Destroy();
}

} // namespace

TEST(VDB, Handle) {
  TestVDBHandle();
}

} // namespace ARIA
