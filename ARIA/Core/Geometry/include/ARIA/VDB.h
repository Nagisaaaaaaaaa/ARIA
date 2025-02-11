#pragma once

/// \file
/// \brief A light-weighted policy-based `VDB` implementation.
/// See https://www.openvdb.org/ if you are not familiar with "vdb".
///
/// Compared with `openvdb` and `nanovdb`, even though this one is not that fast,
/// it supports dynamic memory allocations, even for GPU memory, which
/// makes it much easier to handle dynamic cases.
//
//
//
//
//
#include "ARIA/detail/VDBImpl.h"

namespace ARIA {

/// \brief A light-weighted policy-based `VDB` implementation.
/// See https://www.openvdb.org/ if you are not familiar with "vdb".
///
/// Compared with `openvdb` and `nanovdb`, even though this one is not that fast,
/// it supports dynamic memory allocations, even for GPU memory, which
/// makes it much easier to handle dynamic cases.
///
/// \example ```cpp
/// // Define the volume type and the coordinate type.
/// using Volume = DeviceVDB<float, 2>;
/// using VTec = Tec<int, int>;
///
/// // Instantiate a volume.
/// Volume volume;
///
/// // Instantiate a layout, whose size is 200 * 300.
/// const Layout layout = make_layout_major(200, 300);
///
/// // Get the `VDB` accessors, there are 3 types.
/// VDBAccessor allocateWriteAccessor = volume.allocateWriteAccessor();
/// VDBAccessor writeAccessor = volume.writeAccessor();
/// VDBAccessor readAccessor = volume.readAccessor();
///
/// // Launch for each coord in the layout.
/// Launcher(layout, [=] ARIA_DEVICE(const VTec &tec) mutable {
///   // An `AllocateWriteAccessor` will automatically allocate memory when
///   // value of an unallocated coord is set.
///   allocateWriteAccessor.value(tec) = static_cast<float>(layout(tec));
/// }).Launch();
///
/// // Launch for each coord in the VDB whose value is "on".
/// Launcher(volume, [=] ARIA_DEVICE(const VTec &tec) mutable {
///   // A `WriteAccessor` assumes the memory of the value has always been allocated before.
///   writeAccessor.value(tec) *= 2;
/// }).Launch();
///
/// Launcher(volume, [=] ARIA_DEVICE(const VTec &tec) {
///   // A `ReadAccessor` can only get and cannot set the values.
///   ARIA_ASSERT(readAccessor.value(tec) == layout(tec) * 2);
/// }).Launch();
///
/// // Set some values to "off".
/// Launcher(volume, [=] ARIA_DEVICE(const VTec &tec) mutable {
///   if (get<0>(tec) % 2 == 0)
///     writeAccessor.value(tec) = Off{}; // This will set the value at `tec` to "off".
/// }).Launch();
///
/// // Check whether the values are "on" or "off".
/// Launcher(volume, [=] ARIA_DEVICE(const VTec &tec) {
///   if (get<0>(tec) % 2 == 0)
///     ARIA_ASSERT(!readAccessor.IsValueOn(tec)); // Value at this `tec` should be "off".
///   else
///     ARIA_ASSERT(readAccessor.IsValueOn(tec)); // Value at this `tec` should be "on".
/// }).Launch();
///
/// // After setting some values to "off", you can shrink to fit the `VDB` to save memory.
/// volume.ShrinkToFit();
///
/// // The `Launcher` can also automatically create accessors for each thread.
/// // These `Launcher`-generated accessors are initialized with caches (like openvdb and nanovdb), which
/// // contain information about the per-thread `tec`.
/// Launcher(volume, [=] ARIA_DEVICE(const VTec &tec, AllocateWriteAccessor& accessor) { ... }).Launch();
/// Launcher(volume, [=] ARIA_DEVICE(const VTec &tec, WriteAccessor& accessor) { ... }).Launch();
/// Launcher(volume, [=] ARIA_DEVICE(const VTec &tec, const ReadAccessor& accessor) { ... }).Launch();
/// ```
///
/// \note `VDB` supports both array of structures (AoS) and structure of arrays (SoA).
/// See `Mosaic.h`, `Array.h`, and `Vector.h` for more details.
///
/// \todo `HostVDB`s have not been implemented yet.
using vdb::detail::VDB;

/// \brief A `HostVDB` is a `VDB` whose values are contained in host storages, and
/// CPU accesses are allowed.
///
/// \example ```cpp
/// using Volume = HostVDB<float, 2>;
/// Volume volume;
/// ```
template <typename T, auto dim>
using HostVDB = VDB<T, dim, SpaceHost>;

/// \brief A `DeviceVDB` is a `VDB` whose values are contained in device storages, and
/// GPU accesses are allowed.
///
/// \example ```cpp
/// using Volume = DeviceVDB<float, 2>;
/// Volume volume;
/// ```
template <typename T, auto dim>
using DeviceVDB = VDB<T, dim, SpaceDevice>;

//
//
//
/// \brief A `VDBAccessor` is a host or device view of a `VDB`.
///
/// \example ```cpp
/// VDBAccessor allocateWriteAccessor = volume.allocateWriteAccessor();
/// VDBAccessor writeAccessor = volume.writeAccessor();
/// VDBAccessor readAccessor = volume.readAccessor();
/// ```
using vdb::detail::VDBAccessor;

/// \brief A `VDBAllocateWriteAccessor` will automatically allocate memory when
/// value of an unallocated coord is set.
///
/// \example ```cpp
/// VDBAllocateWriteAccessor allocateWriteAccessor = volume.allocateWriteAccessor();
/// ```
template <typename VDB>
using VDBAllocateWriteAccessor = typename VDB::AllocateWriteAccessor;

/// \brief A `VDBWriteAccessor` assumes the memory of the value has always been allocated before.
///
/// \example ```cpp
/// VDBWriteAccessor writeAccessor = volume.writeAccessor();
/// ```
template <typename VDB>
using VDBWriteAccessor = typename VDB::WriteAccessor;

/// \brief A `VDBReadAccessor` can only get and cannot set the values.
///
/// \example ```cpp
/// VDBReadAccessor readAccessor = volume.readAccessor();
/// ```
template <typename VDB>
using VDBReadAccessor = typename VDB::ReadAccessor;

//
//
//
/// \brief Launch a functor or a lambda function with
/// all the coordinates of a `VDB`, whose values are currently "on".
///
/// \example ```cpp
/// using Volume = DeviceVDB<float, 2>;
/// using VTec = Tec<int, int>;
///
/// Volume volume;
/// ...
///
/// Launcher(volume, [=, accessor = volume.writeAccessor()] ARIA_DEVICE(const VTec &tec) mutable {
///   accessor.value(tec) *= 2;
/// }).Launch();
///
/// Launcher(volume, [=, accessor = volume.readAccessor()] ARIA_DEVICE(const VTec &tec) {
///   ARIA_ASSERT(accessor.value(tec) == layout(tec) * 2);
/// }).Launch();
/// ```
template <vdb::detail::DeviceVDBType TVDB, typename F>
class Launcher<TVDB, F> : public Launcher<typename TVDB::THandle, F> {
private:
  using Base = Launcher<typename TVDB::THandle, F>;

public:
  Launcher(const TVDB &vdb, const F &f) : Base(*vdb.handle_, f) {}

  ARIA_COPY_MOVE_ABILITY(Launcher, default, default);
};

template <vdb::detail::DeviceVDBType TVDB, typename F>
Launcher(const TVDB &vdb, const F &f) -> Launcher<TVDB, F>;

} // namespace ARIA
