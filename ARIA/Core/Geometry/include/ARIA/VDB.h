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

// Device VDB accessor.
template <typename T, auto dim>
class VDBAccessor<T, dim, SpaceDevice> {
public:
  VDBAccessor() : blocks_(TBlocks::createDeviceObject(nBlocksMax)) {}

  ARIA_COPY_MOVE_ABILITY(VDBAccessor, delete, default);

  ~VDBAccessor() noexcept /* Actually, exceptions may be thrown here. */ { TBlocks::destroyDeviceObject(blocks_); }

  //
  //
  //
public:
  // The maximum number of blocks.
  static constexpr size_t nBlocksMax = 1024; // TODO: Maybe still too small, who knows.

  // Number of cells per dim of each block.
  // Eg: dim: 1    1 << dim: 2    512 / (1 << dim): 256    #cells per block: 256
  //          2              4                      128                      16384
  //          3              8                      64                       262144
  //          4              16                     32                       1048576
  static constexpr int nCellsPerBlockDim = 512 / (1 << dim);
  static_assert(nCellsPerBlockDim > 0, "The given dimension is too large");

  // Number of cells per block.
  static constexpr int nCellsPerBlock = vdb::detail::powN<dim>(nCellsPerBlockDim);

  //
  //
  //
  // Type of the coordinate.
  using TCoord = Vec<int, dim>;

  // Type of the space filling curve encoder and decoder used to hash the block coord to and from the block index.
  using TCode = MortonCode<dim>;

  // Type of the block storage which contains whether each cell is on or off.
  using TOnOffBlockStorage = BitArray<nCellsPerBlock, ThreadSafe>;

  // Type of the block storage which contains the actual value of each cell.
  using TDataBlockStroage = cuda::std::array<T, nCellsPerBlock>;

  // Type of the block storage which contains all things of a block.
  struct TBlockStorage {
    TOnOffBlockStorage onOff;
    TDataBlockStroage data;
  };

  // Type of the block.
  class TBlock {
  public:
    TBlockStorage *p = nullptr;

    ARIA_HOST_DEVICE inline void wait() noexcept {
      cuda::std::atomic_ref lockRef{lock_};

      // Wait until the lock is free.
      while (!lockRef.load(cuda::std::memory_order_acquire)) {
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

    ARIA_HOST_DEVICE inline void go() noexcept {
      cuda::std::atomic_ref lockRef{lock_};

      lockRef.store(0, cuda::std::memory_order_release);
    }

  private:
    uint lock_ = 1;
  };

  // Type of the sparse blocks tree.
  using TBlocks = stdgpu::unordered_map<uint64, TBlock>;

  TBlocks blocks_;

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
    constexpr auto n = nCellsPerBlockDim; // TODO: Compiler bug here.
    return cellCoord / n;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 CellCoord2BlockIdx(const TCoord &cellCoord) {
    return BlockCoord2BlockIdx(CellCoord2BlockCoord(cellCoord));
  }

public:
  ARIA_HOST_DEVICE void Allocate(const TCoord &cellCoord) {
    auto res = blocks_.emplace(CellCoord2BlockIdx(cellCoord), TBlock{});

    if (res.second) {
      TBlock &block = res.first->second;

      block.p = new TBlockStorage();

      block.go();
    } else {
      TBlock &block = blocks_.find(CellCoord2BlockIdx(cellCoord))->second;

      // Wait until the block is allocated.
      block.wait();
    }
  }
};

//
//
//
// Device VDB.
template <typename T, auto dim>
class VDB<T, dim, SpaceDevice> {
public:
private:
};

} // namespace ARIA
