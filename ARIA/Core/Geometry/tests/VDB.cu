#include "ARIA/Launcher.h"
#include "ARIA/VDB.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

void TestVDBHandle() {
  using Handle = VDBHandle<float, 2, SpaceDevice>;

  Handle handle = Handle::Create();

  const size_t n = 10000;

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { handle.value_AllocateIfNotExist({i, 0}) = i; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable {
    ARIA_ASSERT(handle.value_AllocateIfNotExist({i, 0}) == i);
  }).Launch();

  cuda::device::current::get().synchronize();

  handle.Destroy();
}

void TestVDBAccessor() {
  using Handle = VDBHandle<float, 2, SpaceDevice>;
  using AllocateWriteAccessor = VDBAllocateWriteAccessor<float, 2, SpaceDevice>;
  using WriteAccessor = VDBWriteAccessor<float, 2, SpaceDevice>;
  using ReadAccessor = VDBReadAccessor<float, 2, SpaceDevice>;

  Handle handle = Handle::Create();
  AllocateWriteAccessor allocateWriteAccessor{handle};
  WriteAccessor writeAccessor{handle};
  ReadAccessor readAccessor{handle};

  const size_t n = 10000;

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { allocateWriteAccessor.value({i, 0}) = i; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { ARIA_ASSERT(allocateWriteAccessor.value({i, 0}) == i); }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { writeAccessor.value({i, 0}) = -i; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { ARIA_ASSERT(writeAccessor.value({i, 0}) == -i); }).Launch();

  // Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { readAccessor.value({i, 0}) = -i; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { ARIA_ASSERT(readAccessor.value({i, 0}) == -i); }).Launch();

  cuda::device::current::get().synchronize();

  handle.Destroy();
}

} // namespace

TEST(VDB, Handle) {
  size_t size = 1LLU * 1024LLU * 1024LLU * 1024LLU; // 1GB
  EXPECT_EQ(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size), cudaSuccess);

  TestVDBHandle();
  TestVDBAccessor();
}

} // namespace ARIA
