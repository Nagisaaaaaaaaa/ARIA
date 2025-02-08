#include "ARIA/VDB.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

template <typename T_, typename U_>
ARIA_HOST_DEVICE inline void AssertEq(const T_ &a_, const U_ &b_) {
  auto a = Auto(a_);
  auto b = Auto(b_);
  using T = decltype(a);
  using U = decltype(b);

  if constexpr (std::integral<T> && std::integral<U>) {
    ARIA_ASSERT(a == b);
  } else {
    ARIA_ASSERT(std::abs(a - b) < 1e-6);
  }
}

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
      AssertEq(handle.value_AllocateIfNotExist(Vec1i{i - nHalf}), i - nHalf);
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
      AssertEq(handle.value_AssumeExist(Vec1i{i - nHalf} * 2), nHalf - i);
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
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      handle.value_AllocateIfNotExist(ToVec(tec)) = layout(tec);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      AssertEq(handle.value_AllocateIfNotExist(ToVec(tec)), layout(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Checkerboard accesses.
  {
    Handle handle = Handle::Create();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec)) % 2 == 0)
        handle.value_AllocateIfNotExist(ToVec(tec)) = layout(tec);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec)) % 2 == 0)
        AssertEq(handle.value_AssumeExist(ToVec(tec)), layout(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Sparse accesses, 1D.
  { // x.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist(Vec2i{i - nHalf, 0}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(handle.value_AllocateIfNotExist(Vec2i{i - nHalf, 0}), i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }
  { // y.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist(Vec2i{0, nHalf - i}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(handle.value_AssumeExist(Vec2i{0, nHalf - i}), i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Sparse accesses, 2D.
  {
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist(Vec2i{i - nHalf, i - nHalf}) = i - nHalf;
      handle.value_AllocateIfNotExist(Vec2i{i - nHalf, nHalf - i}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(handle.value_AllocateIfNotExist(Vec2i{i - nHalf, i - nHalf}), i - nHalf);
      AssertEq(handle.value_AssumeExist(Vec2i{i - nHalf, nHalf - i}), i - nHalf);
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
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      handle.value_AllocateIfNotExist(ToVec(tec)) = layout(tec);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      AssertEq(handle.value_AllocateIfNotExist(ToVec(tec)), layout(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Checkerboard accesses.
  {
    Handle handle = Handle::Create();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0)
        handle.value_AllocateIfNotExist(ToVec(tec)) = layout(tec);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0)
        AssertEq(handle.value_AssumeExist(ToVec(tec)), layout(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Sparse accesses, 1D.
  { // x.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist(Vec3i{i - nHalf, 0, 0}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(handle.value_AllocateIfNotExist(Vec3i{i - nHalf, 0, 0}), i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }
  { // y.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist(Vec3i{0, nHalf - i, 0}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(handle.value_AssumeExist(Vec3i{0, nHalf - i, 0}), i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }
  { // z.
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist(Vec3i{0, 0, nHalf - i}) = i - nHalf;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(handle.value_AssumeExist(Vec3i{0, 0, nHalf - i}), i - nHalf);
    }).Launch();
    cuda::device::current::get().synchronize();
    handle.Destroy();
  }

  // Sparse accesses, 3D.
  {
    Handle handle = Handle::Create();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      handle.value_AllocateIfNotExist(Vec3i{i - nHalf, i - nHalf, 0}) = i - nHalf;
      handle.value_AllocateIfNotExist(Vec3i{i - nHalf, nHalf - i, 0}) = i - nHalf;

      if (i - nHalf != nHalf - i) {
        handle.value_AllocateIfNotExist(Vec3i{i - nHalf, 0, i - nHalf}) = (i - nHalf) * 2;
        handle.value_AllocateIfNotExist(Vec3i{i - nHalf, 0, nHalf - i}) = (i - nHalf) * 2;
      }

      if (i - nHalf != nHalf - i) {
        handle.value_AllocateIfNotExist(Vec3i{0, i - nHalf, i - nHalf}) = (i - nHalf) * (-3);
        handle.value_AllocateIfNotExist(Vec3i{0, i - nHalf, nHalf - i}) = (i - nHalf) * (-3);
      }
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(handle.value_AllocateIfNotExist(Vec3i{i - nHalf, i - nHalf, 0}), i - nHalf);
      AssertEq(handle.value_AssumeExist(Vec3i{i - nHalf, nHalf - i, 0}), i - nHalf);

      if (i - nHalf != nHalf - i) {
        AssertEq(handle.value_AllocateIfNotExist(Vec3i{i - nHalf, 0, i - nHalf}), (i - nHalf) * 2);
        AssertEq(handle.value_AssumeExist(Vec3i{i - nHalf, 0, nHalf - i}), (i - nHalf) * 2);
      }

      if (i - nHalf != nHalf - i) {
        AssertEq(handle.value_AllocateIfNotExist(Vec3i{0, i - nHalf, i - nHalf}), (i - nHalf) * (-3));
        AssertEq(handle.value_AssumeExist(Vec3i{0, i - nHalf, nHalf - i}), (i - nHalf) * (-3));
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

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { allocateWriteAccessor.value(Tec{i - nHalf}) = i - nHalf; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(allocateWriteAccessor.value(Tec{i - nHalf}), i - nHalf);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value(Tec{i - nHalf}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value(Tec{i - nHalf}), (i - nHalf) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(readAccessor.value(Tec{i - nHalf}), (i - nHalf) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { writeAccessor.value(tec) += 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable {
      AssertEq(readAccessor.value(tec), get<0>(tec) * (-2) + 233);
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec, AllocateWriteAccessor accessor) mutable {
      accessor.value(tec) += 233;
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec, ReadAccessor accessor) mutable {
      AssertEq(accessor.value(tec), get<0>(tec) * (-2) + 466);
    }).Launch();
  }

  // Checkerboard accesses.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      allocateWriteAccessor.value(Tec{i - nHalf} * 2) = nHalf - i;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(allocateWriteAccessor.value(Tec{i - nHalf} * 2), nHalf - i);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value(Tec{i - nHalf} * 2) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value(Tec{i - nHalf} * 2), (nHalf - i) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(readAccessor.value(Tec{i - nHalf} * 2), (nHalf - i) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { writeAccessor.value(tec) -= 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable {
      AssertEq(readAccessor.value(tec), get<0>(tec) - 233);
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec, WriteAccessor &accessor) mutable {
      accessor.value(tec) -= 233;
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec, const ReadAccessor &accessor) mutable {
      AssertEq(accessor.value(tec), get<0>(tec) - 466);
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

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      allocateWriteAccessor.value(tec) = layout(tec);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      AssertEq(allocateWriteAccessor.value(tec), layout(tec));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { writeAccessor.value(tec) *= -1; }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      AssertEq(writeAccessor.value(tec), -layout(tec));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), -layout(tec));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { writeAccessor.value(tec) += 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), -layout(tec) + 233);
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec, AllocateWriteAccessor accessor) mutable {
      accessor.value(tec) += 233;
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec, ReadAccessor accessor) mutable {
      AssertEq(accessor.value(tec), -layout(tec) + 466);
    }).Launch();
  }

  // Checkerboard accesses.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec)) % 2 == 0)
        allocateWriteAccessor.value(tec) = layout(tec);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec)) % 2 == 0)
        AssertEq(allocateWriteAccessor.value(tec), layout(tec));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec)) % 2 == 0)
        writeAccessor.value(tec) *= -1;
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec)) % 2 == 0)
        AssertEq(writeAccessor.value(tec), -layout(tec));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec)) % 2 == 0)
        AssertEq(readAccessor.value(tec), -layout(tec));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { writeAccessor.value(tec) += 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), -layout(tec) + 233);
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec, AllocateWriteAccessor &accessor) mutable {
      accessor.value(tec) += 233;
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec, const ReadAccessor &accessor) mutable {
      AssertEq(accessor.value(tec), -layout(tec) + 466);
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
      AssertEq(allocateWriteAccessor.value({i - nHalf, 0}), i - nHalf);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({i - nHalf, 0}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value({i - nHalf, 0}), (i - nHalf) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(readAccessor.value({i - nHalf, 0}), (i - nHalf) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { writeAccessor.value(tec) *= (-3); }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), get<0>(tec) * 6);
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec, WriteAccessor accessor) mutable {
      accessor.value(tec) *= (-3);
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec, ReadAccessor accessor) mutable {
      AssertEq(accessor.value(tec), get<0>(tec) * (-18));
    }).Launch();
  }
  { // y.
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { allocateWriteAccessor.value({0, i - nHalf}) = nHalf - i; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(allocateWriteAccessor.value({0, i - nHalf}), nHalf - i);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({0, i - nHalf}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value({0, i - nHalf}), (nHalf - i) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(readAccessor.value({0, i - nHalf}), (nHalf - i) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { writeAccessor.value(tec) *= (-3); }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), get<1>(tec) * (-6));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec, WriteAccessor &accessor) mutable {
      accessor.value(tec) *= (-3);
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec, ReadAccessor &accessor) mutable {
      AssertEq(accessor.value(tec), get<1>(tec) * 18);
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
      AssertEq(allocateWriteAccessor.value({i - nHalf, i - nHalf}), i - nHalf);
      AssertEq(allocateWriteAccessor.value({i - nHalf, nHalf - i}), i - nHalf);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      writeAccessor.value({i - nHalf, i - nHalf}) *= -3;
      if (i - nHalf != nHalf - i)
        writeAccessor.value({i - nHalf, nHalf - i}) *= -3;
    }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value({i - nHalf, i - nHalf}), (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({i - nHalf, nHalf - i}), (i - nHalf) * (-3));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(readAccessor.value({i - nHalf, i - nHalf}), (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        AssertEq(readAccessor.value({i - nHalf, nHalf - i}), (i - nHalf) * (-3));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { writeAccessor.value(tec) *= (-2); }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(readAccessor.value({i - nHalf, i - nHalf}), (i - nHalf) * 6);
      if (i - nHalf != nHalf - i)
        AssertEq(readAccessor.value({i - nHalf, nHalf - i}), (i - nHalf) * 6);
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

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      allocateWriteAccessor.value(tec) = layout(tec);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      AssertEq(allocateWriteAccessor.value(tec), layout(tec));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      writeAccessor.value(tec) *= -1;
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      AssertEq(writeAccessor.value(tec), -layout(tec));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), -layout(tec));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { writeAccessor.value(tec) += 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), -layout(tec) + 233);
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, AllocateWriteAccessor accessor) mutable {
      accessor.value(tec) += 233;
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, ReadAccessor accessor) mutable {
      AssertEq(accessor.value(tec), -layout(tec) + 466);
    }).Launch();
  }

  // Checkerboard accesses.
  {
    V v;
    VDBAccessor allocateWriteAccessor = v.allocateWriteAccessor();
    VDBAccessor writeAccessor = v.writeAccessor();
    VDBAccessor readAccessor = v.readAccessor();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0)
        allocateWriteAccessor.value(tec) = layout(tec);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0)
        AssertEq(allocateWriteAccessor.value(tec), layout(tec));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0)
        writeAccessor.value(tec) *= -1;
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0)
        AssertEq(writeAccessor.value(tec), -layout(tec));
    }).Launch();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0)
        AssertEq(readAccessor.value(tec), -layout(tec));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { writeAccessor.value(tec) += 233; }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), -layout(tec) + 233);
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, AllocateWriteAccessor &accessor) mutable {
      accessor.value(tec) += 233;
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, const ReadAccessor &accessor) mutable {
      AssertEq(accessor.value(tec), -layout(tec) + 466);
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
      AssertEq(allocateWriteAccessor.value({i - nHalf, 0, 0}), i - nHalf);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({i - nHalf, 0, 0}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value({i - nHalf, 0, 0}), (i - nHalf) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(readAccessor.value({i - nHalf, 0, 0}), (i - nHalf) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { writeAccessor.value(tec) *= (-3); }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), get<0>(tec) * 6);
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, WriteAccessor accessor) mutable {
      accessor.value(tec) *= (-3);
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, ReadAccessor accessor) mutable {
      AssertEq(accessor.value(tec), get<0>(tec) * (-18));
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
      AssertEq(allocateWriteAccessor.value({0, i - nHalf, 0}), nHalf - i);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({0, i - nHalf, 0}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value({0, i - nHalf, 0}), (nHalf - i) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(readAccessor.value({0, i - nHalf, 0}), (nHalf - i) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { writeAccessor.value(tec) *= (-3); }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), get<1>(tec) * (-6));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, WriteAccessor &accessor) mutable {
      accessor.value(tec) *= (-3);
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, ReadAccessor &accessor) mutable {
      AssertEq(accessor.value(tec), get<1>(tec) * 18);
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
      AssertEq(allocateWriteAccessor.value({0, 0, i - nHalf}), nHalf - i);
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable { writeAccessor.value({0, 0, i - nHalf}) *= -2; }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value({0, 0, i - nHalf}), (nHalf - i) * (-2));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(readAccessor.value({0, 0, i - nHalf}), (nHalf - i) * (-2));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { writeAccessor.value(tec) *= (-3); }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      AssertEq(readAccessor.value(tec), get<2>(tec) * (-6));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, AllocateWriteAccessor &accessor) mutable {
      accessor.value(tec) *= (-3);
    }).Launch();
    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec, const AllocateWriteAccessor &accessor) mutable {
      AssertEq(accessor.value(tec), get<2>(tec) * 18);
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
      AssertEq(allocateWriteAccessor.value({i - nHalf, i - nHalf, 0}), i - nHalf);
      AssertEq(allocateWriteAccessor.value({i - nHalf, nHalf - i, 0}), i - nHalf);

      if (i - nHalf != nHalf - i) {
        AssertEq(allocateWriteAccessor.value({i - nHalf, 0, i - nHalf}), (i - nHalf) * 2);
        AssertEq(allocateWriteAccessor.value({i - nHalf, 0, nHalf - i}), (i - nHalf) * 2);
      }

      if (i - nHalf != nHalf - i) {
        AssertEq(allocateWriteAccessor.value({0, i - nHalf, i - nHalf}), (i - nHalf) * (-3));
        AssertEq(allocateWriteAccessor.value({0, i - nHalf, nHalf - i}), (i - nHalf) * (-3));
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
      AssertEq(writeAccessor.value({i - nHalf, i - nHalf, 0}), (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({i - nHalf, nHalf - i, 0}), (i - nHalf) * (-3));

      AssertEq(writeAccessor.value({i - nHalf, 0, i - nHalf}), (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({i - nHalf, 0, nHalf - i}), (i - nHalf) * (-3));

      AssertEq(writeAccessor.value({0, i - nHalf, i - nHalf}), (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({0, i - nHalf, nHalf - i}), (i - nHalf) * (-3));
    }).Launch();

    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value({i - nHalf, i - nHalf, 0}), (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({i - nHalf, nHalf - i, 0}), (i - nHalf) * (-3));

      AssertEq(writeAccessor.value({i - nHalf, 0, i - nHalf}), (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({i - nHalf, 0, nHalf - i}), (i - nHalf) * (-3));

      AssertEq(writeAccessor.value({0, i - nHalf, i - nHalf}), (i - nHalf) * (-3));
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({0, i - nHalf, nHalf - i}), (i - nHalf) * (-3));
    }).Launch();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { writeAccessor.value(tec) *= (-2); }).Launch();
    Launcher(n, [=] ARIA_DEVICE(int i) mutable {
      AssertEq(writeAccessor.value({i - nHalf, i - nHalf, 0}), (i - nHalf) * 6);
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({i - nHalf, nHalf - i, 0}), (i - nHalf) * 6);

      AssertEq(writeAccessor.value({i - nHalf, 0, i - nHalf}), (i - nHalf) * 6);
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({i - nHalf, 0, nHalf - i}), (i - nHalf) * 6);

      AssertEq(writeAccessor.value({0, i - nHalf, i - nHalf}), (i - nHalf) * 6);
      if (i - nHalf != nHalf - i)
        AssertEq(writeAccessor.value({0, i - nHalf, nHalf - i}), (i - nHalf) * 6);
    }).Launch();
  }

  cuda::device::current::get().synchronize();
}

void Test1DVDBSetOffAndShrinkKernels() {
  using V = DeviceVDB<int, 1>;

  const Layout layout = make_layout_major(20000);
  const int n = size(layout);

  thrust::device_vector<int> counterD(1);
  thrust::device_ptr counter = counterD.data();

  // Dense off accesses.
  {
    V v;

    VDBAccessor accessor = v.allocateWriteAccessor();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { ARIA_ASSERT(!accessor.IsValueOn(tec)); }).Launch();
    v.ShrinkToFit();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { accessor.value(tec) = 0; }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { ARIA_ASSERT(accessor.IsValueOn(tec)); }).Launch();
    v.ShrinkToFit();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable {
      accessor.value(tec) = Off{};
      atomicAdd(counter.get(), 1);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { ARIA_ASSERT(!accessor.IsValueOn(tec)); }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { accessor.value(tec) = Off{}; }).Launch();
    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }
  }

  // Checkerboard accesses.
  {
    V v;

    VDBAccessor accessor = v.allocateWriteAccessor();
    v.ShrinkToFit();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { accessor.value(tec) = 0; }).Launch();
    v.ShrinkToFit();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable {
      if (get<0>(tec) % 2 == 0) {
        accessor.value(tec) = Off{};
        atomicAdd(counter.get(), 1);
      }
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int> &tec) mutable {
      if (get<0>(tec) % 2 == 0)
        ARIA_ASSERT(!accessor.IsValueOn(tec));
      else
        ARIA_ASSERT(accessor.IsValueOn(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n / 2);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable {
        if (get<0>(tec) % 2 == 0)
          ARIA_ASSERT(false);
        else
          atomicAdd(counter.get(), 1);
      }).Launch();
      cuda::device::current::get().synchronize();
      EXPECT_EQ(*counter, n / 2);
      *counter = 0;
      v.ShrinkToFit();
    }

    Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable {
      accessor.value(tec) = Off{};
      atomicAdd(counter.get(), 1);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { ARIA_ASSERT(!accessor.IsValueOn(tec)); }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n / 2);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int> &tec) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }
  }
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
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      ARIA_ASSERT(!accessor.IsValueOn(tec));
    }).Launch();
    v.ShrinkToFit();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { accessor.value(tec) = 0; }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      ARIA_ASSERT(accessor.IsValueOn(tec));
    }).Launch();
    v.ShrinkToFit();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      accessor.value(tec) = Off{};
      atomicAdd(counter.get(), 1);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      ARIA_ASSERT(!accessor.IsValueOn(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { accessor.value(tec) = Off{}; }).Launch();
    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }
  }

  // Checkerboard accesses.
  {
    V v;

    VDBAccessor accessor = v.allocateWriteAccessor();
    v.ShrinkToFit();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { accessor.value(tec) = 0; }).Launch();
    v.ShrinkToFit();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec)) % 2 == 0) {
        accessor.value(tec) = Off{};
        atomicAdd(counter.get(), 1);
      }
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec)) % 2 == 0)
        ARIA_ASSERT(!accessor.IsValueOn(tec));
      else
        ARIA_ASSERT(accessor.IsValueOn(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n / 2);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
        if ((get<0>(tec) + get<1>(tec)) % 2 == 0)
          ARIA_ASSERT(false);
        else
          atomicAdd(counter.get(), 1);
      }).Launch();
      cuda::device::current::get().synchronize();
      EXPECT_EQ(*counter, n / 2);
      *counter = 0;
      v.ShrinkToFit();
    }

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      accessor.value(tec) = Off{};
      atomicAdd(counter.get(), 1);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable {
      ARIA_ASSERT(!accessor.IsValueOn(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n / 2);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int, int> &tec) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }
  }
}

void Test3DVDBSetOffAndShrinkKernels() {
  using V = DeviceVDB<int, 3>;

  const Layout layout = make_layout_major(50, 100, 150);
  const int n = size(layout);

  thrust::device_vector<int> counterD(1);
  thrust::device_ptr counter = counterD.data();

  // Dense off accesses.
  {
    V v;

    VDBAccessor accessor = v.allocateWriteAccessor();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      ARIA_ASSERT(!accessor.IsValueOn(tec));
    }).Launch();
    v.ShrinkToFit();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { accessor.value(tec) = 0; }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      ARIA_ASSERT(accessor.IsValueOn(tec));
    }).Launch();
    v.ShrinkToFit();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      accessor.value(tec) = Off{};
      atomicAdd(counter.get(), 1);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      ARIA_ASSERT(!accessor.IsValueOn(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { accessor.value(tec) = Off{}; }).Launch();
    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { ARIA_ASSERT(false); }).Launch();
      v.ShrinkToFit();
    }
  }

  // Checkerboard accesses.
  {
    V v;

    VDBAccessor accessor = v.allocateWriteAccessor();
    v.ShrinkToFit();

    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { accessor.value(tec) = 0; }).Launch();
    v.ShrinkToFit();

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0) {
        accessor.value(tec) = Off{};
        atomicAdd(counter.get(), 1);
      }
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0)
        ARIA_ASSERT(!accessor.IsValueOn(tec));
      else
        ARIA_ASSERT(accessor.IsValueOn(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n / 2);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
        if ((get<0>(tec) + get<1>(tec) + get<2>(tec)) % 2 == 0)
          ARIA_ASSERT(false);
        else
          atomicAdd(counter.get(), 1);
      }).Launch();
      cuda::device::current::get().synchronize();
      EXPECT_EQ(*counter, n / 2);
      *counter = 0;
      v.ShrinkToFit();
    }

    Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      accessor.value(tec) = Off{};
      atomicAdd(counter.get(), 1);
    }).Launch();
    Launcher(layout, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable {
      ARIA_ASSERT(!accessor.IsValueOn(tec));
    }).Launch();
    cuda::device::current::get().synchronize();
    EXPECT_EQ(*counter, n / 2);
    *counter = 0;

    for (int round = 0; round < 3; ++round) {
      Launcher(v, [=] ARIA_DEVICE(const Tec<int, int, int> &tec) mutable { ARIA_ASSERT(false); }).Launch();
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
  Test1DVDBSetOffAndShrinkKernels();
  Test2DVDBSetOffAndShrinkKernels();
  Test3DVDBSetOffAndShrinkKernels();
}

} // namespace ARIA
