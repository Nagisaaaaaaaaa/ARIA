#pragma once

/// \file
/// \warning `VDB` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/detail/VDBImpl.h"

namespace ARIA {

/// \brief A light-weighted `VDB` implementation.
/// See https://www.openvdb.org/ if you are not familiar with "vdb".
///
/// For readonly cases, `VDB` is mush slower than `openvdb` and `nanovdb`, but
/// it support dynamic memory allocations, even for GPU memory, which
/// makes it easier to handle dynamic cases.
///
/// \example ```cpp
/// ```
///
/// \todo Support host VDB.
using vdb::detail::VDB;

template <typename T, auto dim>
using HostVDB = VDB<T, dim, SpaceHost>;

template <typename T, auto dim>
using DeviceVDB = VDB<T, dim, SpaceDevice>;

//
//
//
using vdb::detail::VDBAccessor;

template <typename VDB>
using VDBAllocateWriteAccessor = typename VDB::AllocateWriteAccessor;

template <typename VDB>
using VDBWriteAccessor = typename VDB::WriteAccessor;

template <typename VDB>
using VDBReadAccessor = typename VDB::ReadAccessor;

//
//
//
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
