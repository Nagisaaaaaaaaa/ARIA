#include "ARIA/VDB.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

void Test1DVDBHandleKernels() {
  using Handle = vdb::detail::VDBHandle<float, 1, SpaceDevice>;

  const int n = 20000;
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
  {
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

void Test3DVDBHandleKernels() {
  using Handle = vdb::detail::VDBHandle<float, 3, SpaceDevice>;

  const Layout layout = make_layout_major(50, 100, 150);
  const int n = 1000;
  const int nHalf = n / 2;

  // Dense accesses.
  {
    Handle handle = Handle::Create();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      handle.value_AllocateIfNotExist(ToVec(coord)) = layout(coord);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      ARIA_ASSERT(handle.value_AllocateIfNotExist(ToVec(coord)) == layout(coord));
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Checkerboard accesses.
  {
    Handle handle = Handle::Create();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord) + get<2>(coord)) % 2 == 0)
        handle.value_AllocateIfNotExist(ToVec(coord)) = layout(coord);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord) + get<2>(coord)) % 2 == 0)
        ARIA_ASSERT(handle.value_AssumeExist(ToVec(coord)) == layout(coord));
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Sparse accesses, 1D.
  { // x.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist({i - nHalf, 0, 0}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(handle.value_AllocateIfNotExist({i - nHalf, 0, 0}) == i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }
  { // y.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist({0, nHalf - i, 0}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(handle.value_AssumeExist({0, nHalf - i, 0}) == i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }
  { // z.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist({0, 0, nHalf - i}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(handle.value_AssumeExist({0, 0, nHalf - i}) == i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Sparse accesses, 3D.
  {
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist({i - nHalf, i - nHalf, 0}) = i - nHalf;
      handle.value_AllocateIfNotExist({i - nHalf, nHalf - i, 0}) = i - nHalf;

      if (i - nHalf != nHalf - i) {
        handle.value_AllocateIfNotExist({i - nHalf, 0, i - nHalf}) = (i - nHalf) * 2;
        handle.value_AllocateIfNotExist({i - nHalf, 0, nHalf - i}) = (i - nHalf) * 2;
      }

      if (i - nHalf != nHalf - i) {
        handle.value_AllocateIfNotExist({0, i - nHalf, i - nHalf}) = (i - nHalf) * (-3);
        handle.value_AllocateIfNotExist({0, i - nHalf, nHalf - i}) = (i - nHalf) * (-3);
      }
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(handle.value_AllocateIfNotExist({i - nHalf, i - nHalf, 0}) == i - nHalf);
      ARIA_ASSERT(handle.value_AssumeExist({i - nHalf, nHalf - i, 0}) == i - nHalf);

      if (i - nHalf != nHalf - i) {
        ARIA_ASSERT(handle.value_AllocateIfNotExist({i - nHalf, 0, i - nHalf}) == (i - nHalf) * 2);
        ARIA_ASSERT(handle.value_AssumeExist({i - nHalf, 0, nHalf - i}) == (i - nHalf) * 2);
      }

      if (i - nHalf != nHalf - i) {
        ARIA_ASSERT(handle.value_AllocateIfNotExist({0, i - nHalf, i - nHalf}) == (i - nHalf) * (-3));
        ARIA_ASSERT(handle.value_AssumeExist({0, i - nHalf, nHalf - i}) == (i - nHalf) * (-3));
      }
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }
}

void Test1DVDBKernels() {
  using V = DeviceVDB<int, 1>;
  using AllocateWriteAccessor = VDBAllocateWriteAccessor<V>;
  using WriteAccessor = VDBWriteAccessor<V>;
  using ReadAccessor = VDBReadAccessor<V>;

  const int n = 20000;
  const int nHalf = n / 2;

  // CTAD.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    static_assert(std::is_same_v<decltype(allocateWriteAccessor), AllocateWriteAccessor>);
    static_assert(std::is_same_v<decltype(writeAccessor), WriteAccessor>);
    static_assert(std::is_same_v<decltype(readAccessor), ReadAccessor>);
  }

  // Dense accesses.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      allocateWriteAccessor.value(make_coord(i - nHalf)) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value(make_coord(i - nHalf)) == i - nHalf);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value(make_coord(i - nHalf)) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value(make_coord(i - nHalf)) == (i - nHalf) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(readAccessor.value(make_coord(i - nHalf)) == (i - nHalf) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int> &coord) mutable { writeAccessor.value(coord) += 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == get<0>(coord) * (-2) + 233);
    }).Launch();
  }

  // Checkerboard accesses.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      allocateWriteAccessor.value(make_coord(i - nHalf) * 2) = nHalf - i;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value(make_coord(i - nHalf) * 2) == nHalf - i);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value(make_coord(i - nHalf) * 2) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value(make_coord(i - nHalf) * 2) == (nHalf - i) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(readAccessor.value(make_coord(i - nHalf) * 2) == (nHalf - i) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int> &coord) mutable { writeAccessor.value(coord) -= 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == get<0>(coord) - 233);
    }).Launch();
  }

  cuda::device::current::get().synchronize();
}

void Test2DVDBKernels() {
  using V = DeviceVDB<int, 2>;
  using AllocateWriteAccessor = VDBAllocateWriteAccessor<V>;
  using WriteAccessor = VDBWriteAccessor<V>;
  using ReadAccessor = VDBReadAccessor<V>;

  const Layout layout = make_layout_major(200, 300);
  const int n = 20000;
  const int nHalf = n / 2;

  // CTAD.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    static_assert(std::is_same_v<decltype(allocateWriteAccessor), AllocateWriteAccessor>);
    static_assert(std::is_same_v<decltype(writeAccessor), WriteAccessor>);
    static_assert(std::is_same_v<decltype(readAccessor), ReadAccessor>);
  }

  // Dense accesses.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      allocateWriteAccessor.value(coord) = layout(coord);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value(coord) == layout(coord));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      writeAccessor.value(coord) *= -1;
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      ARIA_ASSERT(writeAccessor.value(coord) == -layout(coord));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == -layout(coord));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { writeAccessor.value(coord) += 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == -layout(coord) + 233);
    }).Launch();
  }

  // Checkerboard accesses.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord)) % 2 == 0)
        allocateWriteAccessor.value(coord) = layout(coord);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord)) % 2 == 0)
        ARIA_ASSERT(allocateWriteAccessor.value(coord) == layout(coord));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord)) % 2 == 0)
        writeAccessor.value(coord) *= -1;
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord)) % 2 == 0)
        ARIA_ASSERT(writeAccessor.value(coord) == -layout(coord));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord)) % 2 == 0)
        ARIA_ASSERT(readAccessor.value(coord) == -layout(coord));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { writeAccessor.value(coord) += 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == -layout(coord) + 233);
    }).Launch();
  }

  // Sparse accesses, 1D.
  { // x.
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { allocateWriteAccessor.value({i - nHalf, 0}) = i - nHalf; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value({i - nHalf, 0}) == i - nHalf);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({i - nHalf, 0}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value({i - nHalf, 0}) == (i - nHalf) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(readAccessor.value({i - nHalf, 0}) == (i - nHalf) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { writeAccessor.value(coord) *= (-3); }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == get<0>(coord) * 6);
    }).Launch();
  }
  { // y.
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { allocateWriteAccessor.value({0, i - nHalf}) = nHalf - i; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value({0, i - nHalf}) == nHalf - i);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({0, i - nHalf}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value({0, i - nHalf}) == (nHalf - i) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(readAccessor.value({0, i - nHalf}) == (nHalf - i) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { writeAccessor.value(coord) *= (-3); }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == get<1>(coord) * (-6));
    }).Launch();
  }

  // Sparse accesses, 2D.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      allocateWriteAccessor.value({i - nHalf, i - nHalf}) = i - nHalf;
      allocateWriteAccessor.value({i - nHalf, nHalf - i}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value({i - nHalf, i - nHalf}) == i - nHalf);
      ARIA_ASSERT(allocateWriteAccessor.value({i - nHalf, nHalf - i}) == i - nHalf);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      writeAccessor.value({i - nHalf, i - nHalf}) *= -3;
      if (i - nHalf != nHalf - i)
        writeAccessor.value({i - nHalf, nHalf - i}) *= -3;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value({i - nHalf, i - nHalf}) == (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({i - nHalf, nHalf - i}) == (i - nHalf) * (-3));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(readAccessor.value({i - nHalf, i - nHalf}) == (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(readAccessor.value({i - nHalf, nHalf - i}) == (i - nHalf) * (-3));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { writeAccessor.value(coord) *= (-2); }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(readAccessor.value({i - nHalf, i - nHalf}) == (i - nHalf) * 6);
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(readAccessor.value({i - nHalf, nHalf - i}) == (i - nHalf) * 6);
    }).Launch();
  }

  cuda::device::current::get().synchronize();
}

void Test3DVDBKernels() {
  using V = DeviceVDB<int, 3>;
  using AllocateWriteAccessor = VDBAllocateWriteAccessor<V>;
  using WriteAccessor = VDBWriteAccessor<V>;
  using ReadAccessor = VDBReadAccessor<V>;

  const Layout layout = make_layout_major(50, 100, 150);
  const int n = 1000;
  const int nHalf = n / 2;

  // CTAD.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    static_assert(std::is_same_v<decltype(allocateWriteAccessor), AllocateWriteAccessor>);
    static_assert(std::is_same_v<decltype(writeAccessor), WriteAccessor>);
    static_assert(std::is_same_v<decltype(readAccessor), ReadAccessor>);
  }

  // Dense accesses.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      allocateWriteAccessor.value(coord) = layout(coord);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value(coord) == layout(coord));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      writeAccessor.value(coord) *= -1;
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      ARIA_ASSERT(writeAccessor.value(coord) == -layout(coord));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == -layout(coord));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      writeAccessor.value(coord) += 233;
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == -layout(coord) + 233);
    }).Launch();
  }

  // Checkerboard accesses.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord) + get<2>(coord)) % 2 == 0)
        allocateWriteAccessor.value(coord) = layout(coord);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord) + get<2>(coord)) % 2 == 0)
        ARIA_ASSERT(allocateWriteAccessor.value(coord) == layout(coord));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord) + get<2>(coord)) % 2 == 0)
        writeAccessor.value(coord) *= -1;
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord) + get<2>(coord)) % 2 == 0)
        ARIA_ASSERT(writeAccessor.value(coord) == -layout(coord));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord) + get<2>(coord)) % 2 == 0)
        ARIA_ASSERT(readAccessor.value(coord) == -layout(coord));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      writeAccessor.value(coord) += 233;
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == -layout(coord) + 233);
    }).Launch();
  }

  // Sparse accesses, 1D.
  { // x.
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      allocateWriteAccessor.value({i - nHalf, 0, 0}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value({i - nHalf, 0, 0}) == i - nHalf);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({i - nHalf, 0, 0}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value({i - nHalf, 0, 0}) == (i - nHalf) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(readAccessor.value({i - nHalf, 0, 0}) == (i - nHalf) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      writeAccessor.value(coord) *= (-3);
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == get<0>(coord) * 6);
    }).Launch();
  }
  { // y.
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      allocateWriteAccessor.value({0, i - nHalf, 0}) = nHalf - i;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value({0, i - nHalf, 0}) == nHalf - i);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({0, i - nHalf, 0}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value({0, i - nHalf, 0}) == (nHalf - i) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(readAccessor.value({0, i - nHalf, 0}) == (nHalf - i) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      writeAccessor.value(coord) *= (-3);
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == get<1>(coord) * (-6));
    }).Launch();
  }
  { // z.
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      allocateWriteAccessor.value({0, 0, i - nHalf}) = nHalf - i;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value({0, 0, i - nHalf}) == nHalf - i);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({0, 0, i - nHalf}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value({0, 0, i - nHalf}) == (nHalf - i) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(readAccessor.value({0, 0, i - nHalf}) == (nHalf - i) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      writeAccessor.value(coord) *= (-3);
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      ARIA_ASSERT(readAccessor.value(coord) == get<2>(coord) * (-6));
    }).Launch();
  }

  // Sparse accesses, 3D.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      allocateWriteAccessor.value({i - nHalf, i - nHalf, 0}) = i - nHalf;
      allocateWriteAccessor.value({i - nHalf, nHalf - i, 0}) = i - nHalf;

      if (i - nHalf != nHalf - i) {
        allocateWriteAccessor.value({i - nHalf, 0, i - nHalf}) = (i - nHalf) * 2;
        allocateWriteAccessor.value({i - nHalf, 0, nHalf - i}) = (i - nHalf) * 2;
      }

      if (i - nHalf != nHalf - i) {
        allocateWriteAccessor.value({0, i - nHalf, i - nHalf}) = (i - nHalf) * (-3);
        allocateWriteAccessor.value({0, i - nHalf, nHalf - i}) = (i - nHalf) * (-3);
      }
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(allocateWriteAccessor.value({i - nHalf, i - nHalf, 0}) == i - nHalf);
      ARIA_ASSERT(allocateWriteAccessor.value({i - nHalf, nHalf - i, 0}) == i - nHalf);

      if (i - nHalf != nHalf - i) {
        ARIA_ASSERT(allocateWriteAccessor.value({i - nHalf, 0, i - nHalf}) == (i - nHalf) * 2);
        ARIA_ASSERT(allocateWriteAccessor.value({i - nHalf, 0, nHalf - i}) == (i - nHalf) * 2);
      }

      if (i - nHalf != nHalf - i) {
        ARIA_ASSERT(allocateWriteAccessor.value({0, i - nHalf, i - nHalf}) == (i - nHalf) * (-3));
        ARIA_ASSERT(allocateWriteAccessor.value({0, i - nHalf, nHalf - i}) == (i - nHalf) * (-3));
      }
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      allocateWriteAccessor.value({i - nHalf, i - nHalf, 0}) = i - nHalf;
      allocateWriteAccessor.value({i - nHalf, nHalf - i, 0}) = i - nHalf;

      allocateWriteAccessor.value({i - nHalf, 0, i - nHalf}) = i - nHalf;
      allocateWriteAccessor.value({i - nHalf, 0, nHalf - i}) = i - nHalf;

      allocateWriteAccessor.value({0, i - nHalf, i - nHalf}) = i - nHalf;
      allocateWriteAccessor.value({0, i - nHalf, nHalf - i}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      writeAccessor.value({i - nHalf, i - nHalf, 0}) *= -3;
      if (i - nHalf != nHalf - i)
        writeAccessor.value({i - nHalf, nHalf - i, 0}) *= -3;

      writeAccessor.value({i - nHalf, 0, i - nHalf}) *= -3;
      if (i - nHalf != nHalf - i)
        writeAccessor.value({i - nHalf, 0, nHalf - i}) *= -3;

      writeAccessor.value({0, i - nHalf, i - nHalf}) *= -3;
      if (i - nHalf != nHalf - i)
        writeAccessor.value({0, i - nHalf, nHalf - i}) *= -3;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value({i - nHalf, i - nHalf, 0}) == (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({i - nHalf, nHalf - i, 0}) == (i - nHalf) * (-3));

      ARIA_ASSERT(writeAccessor.value({i - nHalf, 0, i - nHalf}) == (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({i - nHalf, 0, nHalf - i}) == (i - nHalf) * (-3));

      ARIA_ASSERT(writeAccessor.value({0, i - nHalf, i - nHalf}) == (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({0, i - nHalf, nHalf - i}) == (i - nHalf) * (-3));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value({i - nHalf, i - nHalf, 0}) == (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({i - nHalf, nHalf - i, 0}) == (i - nHalf) * (-3));

      ARIA_ASSERT(writeAccessor.value({i - nHalf, 0, i - nHalf}) == (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({i - nHalf, 0, nHalf - i}) == (i - nHalf) * (-3));

      ARIA_ASSERT(writeAccessor.value({0, i - nHalf, i - nHalf}) == (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({0, i - nHalf, nHalf - i}) == (i - nHalf) * (-3));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int, int> &coord) mutable {
      writeAccessor.value(coord) *= (-2);
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      ARIA_ASSERT(writeAccessor.value({i - nHalf, i - nHalf, 0}) == (i - nHalf) * 6);
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({i - nHalf, nHalf - i, 0}) == (i - nHalf) * 6);

      ARIA_ASSERT(writeAccessor.value({i - nHalf, 0, i - nHalf}) == (i - nHalf) * 6);
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({i - nHalf, 0, nHalf - i}) == (i - nHalf) * 6);

      ARIA_ASSERT(writeAccessor.value({0, i - nHalf, i - nHalf}) == (i - nHalf) * 6);
      if (i - nHalf != nHalf - i)
        ARIA_ASSERT(writeAccessor.value({0, i - nHalf, nHalf - i}) == (i - nHalf) * 6);
    }).Launch();
  }

  cuda::device::current::get().synchronize();
}

void Test2DVDBSetOffAndShrinkKernels() {
  using V = DeviceVDB<int, 2>;

  const Layout layout = make_layout_major(200, 300);
  const int n = size(layout);

  thrust::device_vector<int> counterD(1);
  thrust::device_ptr counter = counterD.data();

  // Dense off accesses.
  {
    V v;

    VDBAccessor accessor = v.allocateWriteAccessor();
    v.ShrinkToFit();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { accessor.value(coord) = 0; }).Launch();
    v.ShrinkToFit();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      accessor.value(coord) = Off{};
      atomicAdd(counter.get(), 1);
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { accessor.value(coord) = Off{}; }).Launch();
    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }
  }

  // Checkerboard accesses.
  {
    V v;

    VDBAccessor accessor = v.allocateWriteAccessor();
    v.ShrinkToFit();

    Launcher(layout, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { accessor.value(coord) = 0; }).Launch();
    v.ShrinkToFit();

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      if ((get<0>(coord) + get<1>(coord)) % 2 == 0) {
        accessor.value(coord) = Off{};
        atomicAdd(counter.get(), 1);
      }
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n / 2);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
        if ((get<0>(coord) + get<1>(coord)) % 2 == 0)
          ARIA_ASSERT(false);
        else
          atomicAdd(counter.get(), 1);
      }).Launch();
      cuda::device::current::get().synchronize();
      EXPECT_EQ(*counter, n / 2);
      *counter = 0;
      v.ShrinkToFit();
    }

    Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable {
      accessor.value(coord) = Off{};
      atomicAdd(counter.get(), 1);
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n / 2);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Coord<int, int> &coord) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }
  }
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
  Test3DVDBHandleKernels();
}

TEST(VDB, VDB) {
  Test1DVDBKernels();
  Test2DVDBKernels();
  Test3DVDBKernels();
}

TEST(VDB, SetOffAndShrink) {
  Test2DVDBSetOffAndShrinkKernels();
}

} // namespace ARIA
