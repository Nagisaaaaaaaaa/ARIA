#include "ARIA/VDB.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

void Test1DVDBHandleKernels() {
  using Handle = vdb::detail::VDBHandle<float, 1, SpaceDevice>;

  const int n = 10000;
  const int nHalf = n / 2;

  // Dense accesses.
  {
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist(Vec1i{i - nHalf}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(handle.value_AllocateIfNotExist(Vec1i{i - nHalf}) == i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Checkerboard accesses.
  {
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist(Vec1i{i - nHalf} * 2) = nHalf - i;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(handle.value_AssumeExist(Vec1i{i - nHalf} * 2) == nHalf - i);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }
}

void Test2DVDBHandleKernels() {
  using Handle = vdb::detail::VDBHandle<float, 2, SpaceDevice>;

  const Layout layout = make_layout_major(200, 300);
  const int n = 20000;
  const int nHalf = n / 2;

  // Dense accesses.
  {
    Handle handle = Handle::Create();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      handle.value_AllocateIfNotExist(ToVec(coord)) = layout(coord);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      ARIA_ASSERT(handle.value_AllocateIfNotExist(ToVec(coord)) == layout(coord));
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Checkerboard accesses.
  {
    Handle handle = Handle::Create();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord)) % 2 == 0)
        handle.value_AllocateIfNotExist(ToVec(coord)) = layout(coord);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord)) % 2 == 0)
        ARIA_ASSERT(handle.value_AssumeExist(ToVec(coord)) == layout(coord));
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Sparse accesses, 1D.
  { // x.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist({i - nHalf, 0}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(handle.value_AllocateIfNotExist({i - nHalf, 0}) == i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }
  { // y.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist({0, nHalf - i}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(handle.value_AssumeExist({0, nHalf - i}) == i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Sparse accesses, 2D.
  { // (x, y).
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist({i - nHalf, i - nHalf}) = i - nHalf;
      handle.value_AllocateIfNotExist({i - nHalf, nHalf - i}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(handle.value_AllocateIfNotExist({i - nHalf, i - nHalf}) == i - nHalf);
      ARIA_ASSERT(handle.value_AssumeExist({i - nHalf, nHalf - i}) == i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }
}

void Test2DVDBAccessorsKernels() {
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

  auto testVDBBase = []<auto dim>() {
    using V = DeviceVDB<float, dim>;
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
  };

  testVDBBase.operator()<1>();
  testVDBBase.operator()<2>();
  testVDBBase.operator()<3>();
  testVDBBase.operator()<4>();
}

TEST(VDB, Handle) {
  auto testVDBHandleBase = []<auto dim>() {
    using Handle = vdb::detail::VDBHandle<float, dim, SpaceDevice>;

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
  };

  testVDBHandleBase.operator()<1>();
  testVDBHandleBase.operator()<2>();
  testVDBHandleBase.operator()<3>();
  testVDBHandleBase.operator()<4>();

  Test1DVDBHandleKernels();
  Test2DVDBHandleKernels();
}

TEST(VDB, Accessors) {
  Test2DVDBAccessorsKernels();
}

} // namespace ARIA
