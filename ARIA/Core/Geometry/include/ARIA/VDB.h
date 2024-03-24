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
using vdb::detail::VDB;

//
//
//
template <vdb::detail::DeviceVDBType TVDB, typename F>
class Launcher<TVDB, F> : public launcher::detail::LauncherBase<Launcher<TVDB, F>> {
private:
  using Base = launcher::detail::LauncherBase<Launcher<TVDB, F>>;

public:
  Launcher(const TVDB &vdb, const F &f) : handle_(*vdb.handle_), f_(f) {
    Base::overallSize(cosize_safe_v<TBlockLayout>);
  }

  ARIA_COPY_MOVE_ABILITY(Launcher, default, default);

public:
  using Base::blockSize;

  void Launch() {
    // Shallow-copy blocks from device to host.
    auto blocks = handle_.blocks_.device_range();
    thrust::host_vector<stdgpu::pair<uint64, TBlock>> blocksH(blocks.size());
    thrust::copy(blocks.begin(), blocks.end(), blocksH.begin());

    // For each block.
    for (auto &block : blocksH) {
      // Compute `cellCoordOffset` for this block.
      TVec blockCoord = handle_.BlockIdx2BlockCoord(block.first);
      TVec cellCoordOffset = handle_.BlockCoord2CellCoordOffset(blockCoord);

      // Launch.
      Base::Launch(vdb::detail::KernelLaunchVDBBlock<THandle, F>, block.second, ToCoord(cellCoordOffset), f_);
    }
  }

private:
  using THandle = TVDB::THandle;
  using TVec = THandle::TVec;
  using TBlock = THandle::TBlock;
  using TBlockLayout = THandle::TBlockLayout;

  THandle handle_;
  F f_;
};

template <vdb::detail::DeviceVDBType TVDB, typename F>
Launcher(const TVDB &vdb, const F &f) -> Launcher<TVDB, F>;

} // namespace ARIA
