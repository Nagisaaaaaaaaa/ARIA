#pragma once

/// \file
/// \warning `VDB` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/BitArray.h"
#include "ARIA/ForEach.h"
#include "ARIA/MortonCode.h"
#include "ARIA/Vec.h"

#include <stdgpu/unordered_map.cuh>

namespace ARIA {

template <typename T, auto dim, typename TSpace>
class VDBHandle;

template <typename T, auto dim, typename TSpace>
class VDBAccessor;

template <typename T, auto dim, typename TSpace>
class VDB;

//
//
//
namespace vdb::detail {

template <int N>
ARIA_HOST_DEVICE static int consteval powN(int x) {
  static_assert(N >= 0);

  if constexpr (N > 0) {
    return x * powN<N - 1>(x);
  } else {
    return 1;
  }
}

} // namespace vdb::detail

//
//
//
// Device VDB handle.
template <typename T, auto dim>
class VDBHandle<T, dim, SpaceDevice> {
public:
  VDBHandle() = default;

  ARIA_COPY_MOVE_ABILITY(VDBHandle, default, default);

  [[nodiscard]] static VDBHandle Create() {
    VDBHandle handle;
    handle.blocks_ = TBlocks::createDeviceObject(nBlocksMax);
    return handle;
  }

  void Destroy() noexcept /* Actually, exceptions may be thrown here. */ { TBlocks::destroyDeviceObject(blocks_); }

  //
  //
  //
private:
  // Maximum number of blocks.
  static constexpr size_t nBlocksMax = 512; // TODO: Maybe still too small, who knows.

  // Number of cells per dim of each block.
  // Eg: dim: 1    1 << dim: 2    nCellsPerBlockDim: 256    nCellsPerBlock: 256
  //          2              4                       128                    16384
  //          3              8                       64                     262144
  //          4              16                      32                     1048576
  //          5                                      32                     33554432
  static constexpr int nCellsPerBlockDim = std::max(512 / (1 << dim), 32);

  // Number of cells per block.
  static constexpr int nCellsPerBlock = vdb::detail::powN<dim>(nCellsPerBlockDim); // = nCellsPerBlockDim^dim

  //
  //
  //
  // Type of the coordinate.
  using TCoord = Vec<int, dim>;

  // Type of the space filling curve encoder and decoder, which
  // is used to hash the block coord to and from the block index.
  using TCode = MortonCode<dim>;

  // Type of the block storage part, which contains whether each cell is on or off.
  using TBlockStorageOnOff = BitArray<nCellsPerBlock, ThreadSafe>;

  // Type of the block storage part, which contains the actual value of each cell.
  using TBlockStorageData = cuda::std::array<T, nCellsPerBlock>;

  // Type of the block storage.
  struct TBlockStorage {
    TBlockStorageOnOff onOff;
    TBlockStorageData data;
  };

  // Type of the block, which contains the block storage pointer and a barrier.
  class TBlock {
  public:
    TBlockStorage *p = nullptr;

    // A thread calls this method to mark the storage as ready.
    ARIA_HOST_DEVICE inline void arrive() noexcept {
      cuda::std::atomic_ref barrier{barrier_};

      barrier.store(0, cuda::std::memory_order_release);
    }

    // A thread calls this method to wait for the storage being ready.
    ARIA_HOST_DEVICE inline void wait() noexcept {
      cuda::std::atomic_ref barrier{barrier_};

      // Spin until the barrier is ready.
      while (!barrier.load(cuda::std::memory_order_acquire)) {
#if ARIA_IS_HOST_CODE
  #if ARIA_ICC || ARIA_MSVC
        _mm_pause();
  #else
        __builtin_ia32_pause();
  #endif
#else
        __nanosleep(2);
#endif
      }
    }

  private:
    uint barrier_ = 1;
  };

  // Type of the sparse blocks tree:
  //   Key  : Code of the block coord (defined by TCode).
  //   Value: The block.
  using TBlocks = stdgpu::unordered_map<uint64, TBlock>;

  TBlocks blocks_;

  //
  //
  //
  [[nodiscard]] ARIA_HOST_DEVICE static uint64 BlockCoord2BlockIdx(TCoord blockCoord) {
    // Compute the quadrant.
    uint64 quadrant = 0;
    ForEach<dim>([&]<auto id>() {
      int &axis = blockCoord[id];

      if (axis < 0) {
        axis = -axis;
        quadrant |= (1 << id); // Fill the `id`^th bit.
      }
    });

    // Compute the block index.
    uint64 idx = TCode::Encode(Auto(blockCoord.template cast<uint64>()));

    // Encode the quadrant to the highest bits of the index.
    ARIA_ASSERT((idx & ((~uint64{0}) << (64 - dim))) == 0,
                "The given block coord excesses the representation of the encoder, "
                "please use a larger encoder instead");
    idx |= quadrant << (64 - dim);

    return idx;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static TCoord CellCoord2BlockCoord(const TCoord &cellCoord) {
    // TODO: Compiler bug here: `nCellsPerBlockDim` is not defined in device code.
    // return cellCoord / nCellsPerBlockDim;

    constexpr auto n = nCellsPerBlockDim;
    return cellCoord / n;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 CellCoord2BlockIdx(const TCoord &cellCoord) {
    return BlockCoord2BlockIdx(CellCoord2BlockCoord(cellCoord));
  }

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 CellCoord2CellIdxInBlock(const TCoord &cellCoord) {
    TCoord cellCoordInBlock = cellCoord % nCellsPerBlockDim;
    // TODO: It is better to use CuTe::Layout.
    return TCode::Encode(Auto(cellCoordInBlock.template cast<uint64>()));
  }

private:
  ARIA_DEVICE TBlock &block(const TCoord &cellCoord) {
    // Each thread is trying to insert an empty block into the unordered map,
    // but only one unique thread will succeed.
    auto res = blocks_.emplace(CellCoord2BlockIdx(cellCoord), TBlock{});

    // If success, get reference to the emplaced block.
    TBlock &block = res.first->second;

    if (res.second) { // For the unique thread which succeeded to emplace the block:
      // Allocate the block storage.
      block.p = new TBlockStorage();

      // Mark the storage as ready.
      block.arrive();
    } else { // For other threads which failed:
      // Get reference to the emplaced block.
      block = blocks_.find(CellCoord2BlockIdx(cellCoord))->second;

      // Wait for the storage being ready.
      block.wait();
    }

    // For now, all threads have access to the emplaced block.
    return block;
  }

public:
  ARIA_PROP(public, public, ARIA_HOST_DEVICE, T, value, TCoord);

private:
  [[nodiscard]] ARIA_DEVICE T ARIA_PROP_IMPL(value)(const TCoord &cellCoord) const {
    TBlock &b = block(cellCoord);
    ARIA_ASSERT(b.p->onOff[CellCoord2CellIdxInBlock(cellCoord)]);
    return b.p->data[CellCoord2CellIdxInBlock(cellCoord)];
  }

  ARIA_DEVICE void ARIA_PROP_IMPL(value)(const TCoord &cellCoord, const T &value) {
    TBlock &b = block(cellCoord);
    b.p->data[CellCoord2CellIdxInBlock(cellCoord)] = value;
  }
};

} // namespace ARIA
