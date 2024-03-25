#include "ARIA/VDB.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

void TestVDBHandleKernels() {
  using Handle = vdb::detail::VDBHandle<float, 2, SpaceDevice>;

  Handle handle = Handle::Create();

  const size_t n = 10000;

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { handle.value_AllocateIfNotExist({i, 0}) = i; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable {
    ARIA_ASSERT(handle.value_AllocateIfNotExist({i, 0}) == i);
  }).Launch();

  cuda::device::current::get().synchronize();

  handle.Destroy();
}

void TestVDBAccessorsKernels() {
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

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { writeAccessor.value({i, 0}) *= -1; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { ARIA_ASSERT(writeAccessor.value({i, 0}) == -float(i)); }).Launch();

  // Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { readAccessor.value({i, 0}) *= -1; }).Launch();

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { ARIA_ASSERT(readAccessor.value({i, 0}) == -float(i)); }).Launch();

  Launcher(v, [=] ARIA_DEVICE(const Vec2i &coord) mutable { writeAccessor.value(coord) *= -1; }).Launch();

  Launcher(v, [=] ARIA_DEVICE(const Vec2i &coord) mutable {
    ARIA_ASSERT(writeAccessor.value(coord) == coord.x());
  }).Launch();

  cuda::device::current::get().synchronize();
}

} // namespace

TEST(VDB, Base) {
  size_t size = 1LLU * 1024LLU * 1024LLU * 1024LLU; // 1GB
  cuda::device::current::get().set_limit(CU_LIMIT_MALLOC_HEAP_SIZE, size);

  // Device VDB.
  {
    using V = DeviceVDB<float, 2>;
    using AllocateWriteAccessor = VDBAllocateWriteAccessor<V>;
    using WriteAccessor = VDBWriteAccessor<V>;
    using ReadAccessor = VDBReadAccessor<V>;

    // Constructors.
    V v0{};

    // Move.
    V v = std::move(v0);

    // Allocate-write accessor.
    {
      // Constructors.
      AllocateWriteAccessor accessor0;
      AllocateWriteAccessor accessor1 = v.allocateWriteAccessor();

      // Copy.
      AllocateWriteAccessor accessor2 = accessor0;
      AllocateWriteAccessor accessor3 = accessor1;

      // Move.
      AllocateWriteAccessor accessor4 = std::move(accessor0);
      AllocateWriteAccessor accessor5 = std::move(accessor1);

      // Destructor.
    }

    // Write accessor.
    {
      // Constructors.
      WriteAccessor accessor0;
      WriteAccessor accessor1 = v.writeAccessor();

      // Copy.
      WriteAccessor accessor2 = accessor0;
      WriteAccessor accessor3 = accessor1;

      // Move.
      WriteAccessor accessor4 = std::move(accessor0);
      WriteAccessor accessor5 = std::move(accessor1);

      // Destructor.
    }

    // Read accessor.
    {
      // Constructors.
      ReadAccessor accessor0;
      ReadAccessor accessor1 = v.readAccessor();

      // Copy.
      ReadAccessor accessor2 = accessor0;
      ReadAccessor accessor3 = accessor1;

      // Move.
      ReadAccessor accessor4 = std::move(accessor0);
      ReadAccessor accessor5 = std::move(accessor1);

      // Destructor.
    }

    // Destructor.
  }
}

TEST(VDB, Handle) {
  // Device VDB handle.
  {
    using Handle = vdb::detail::VDBHandle<float, 2, SpaceDevice>;

    // Constructors and create.
    Handle handle0;
    Handle handle1 = Handle::Create();

    // Copy.
    Handle handle2 = handle0;
    Handle handle3 = handle1;

    // Move.
    Handle handle4 = std::move(handle0);
    Handle handle5 = std::move(handle1);

    // Destructor and destroy.
    handle1.Destroy();

    TestVDBHandleKernels();
  }
}

TEST(VDB, Accessors) {
  TestVDBAccessorsKernels();
}

} // namespace ARIA
