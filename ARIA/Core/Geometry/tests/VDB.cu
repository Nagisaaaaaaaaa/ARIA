#include "ARIA/VDB.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

void TestVDBHandleKernels() {
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

void TestVDBKernels() {
  using V = VDB<float, 2, SpaceDevice>;
  using AllocateWriteAccessor = VDBAllocateWriteAccessor<V>;
  using WriteAccessor = VDBWriteAccessor<V>;
  using ReadAccessor = VDBReadAccessor<V>;

  V v;
  VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor(); // Test CTAD.
  VDBAccessor writeAccessor = v.writeAccessor();                 // Test CTAD.
  VDBAccessor readAccessor = v.readAccessor();                   // Test CTAD.

  static_assert(std::is_same_v<decltype(allocateWriteAccessor), AllocateWriteAccessor>);
  static_assert(std::is_same_v<decltype(writeAccessor), WriteAccessor>);
  static_assert(std::is_same_v<decltype(readAccessor), ReadAccessor>);

  const size_t n = 10000;

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { allocateWriteAccessor.value({i, 0}) = i; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { ARIA_ASSERT(allocateWriteAccessor.value({i, 0}) == i); }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { writeAccessor.value({i, 0}) = -i; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { ARIA_ASSERT(writeAccessor.value({i, 0}) == -i); }).Launch();

  // Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { readAccessor.value({i, 0}) = -i; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { ARIA_ASSERT(readAccessor.value({i, 0}) == -i); }).Launch();

  Launcher(v, [=] ARIA_DEVICE(const Vec2i &coord) mutable { writeAccessor.value(coord) *= -1; }).Launch();

  Launcher(v, [=] ARIA_DEVICE(const Vec2i &coord) mutable {
    ARIA_ASSERT(writeAccessor.value(coord) == coord.x());
  }).Launch();

  cuda::device::current::get().synchronize();
}

} // namespace

TEST(VDB, Kernels) {
  size_t size = 1LLU * 1024LLU * 1024LLU * 1024LLU; // 1GB
  cuda::device::current::get().set_limit(CU_LIMIT_MALLOC_HEAP_SIZE, size);

  TestVDBHandleKernels();
  TestVDBKernels();
}

} // namespace ARIA
