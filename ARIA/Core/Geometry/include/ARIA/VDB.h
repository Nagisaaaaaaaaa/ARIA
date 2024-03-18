#pragma once

/// \file
/// \warning `VDB` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/BitVector.h"
#include "ARIA/MortonCode.h"
#include "ARIA/TensorVector.h"
#include "ARIA/Vec.h"

#include <stdgpu/unordered_map.cuh>

#include <bitset>

namespace ARIA {

template <typename T, auto dim, typename TSpace>
class VDBAccessor;

template <typename T, auto dim, typename TSpace>
class VDB;

//
//
//
// Device VDB accessor.
template <typename T, auto dim>
class VDBAccessor<T, dim, SpaceDevice> {
public:
  VDBAccessor() : blocks_(TBlocks::createDeviceObject(nBlocksMax)) {}

  ARIA_COPY_MOVE_ABILITY(VDBAccessor, delete, default);

  // ~VDBAccessor() noexcept /* Actually, exceptions may be thrown here. */ { TBlocks::destroyDeviceObject(blocks_); }

public:
  // TODO: Implement IsValueOn().
  // TODO: Implement GetValue().

private:
  // Type of the coordinate.
  using TCoord = Vec<int, dim>;

  // Type of the space filling curve encoder and decoder used to hash the block coord to and from the block index.
  using TCode = MortonCode<dim>;

  // Type of the block which contains whether each cell is on or off.
  using TOnOffBlock = BitVector<SpaceDevice, ThreadSafe>;

  // Type of the block which contains the actual value of each cell.
  using TDataBlock = thrust::device_vector<T>;

  // Type of the sparse blocks tree.
  using TBlocks = stdgpu::unordered_map<uint64, std::pair<TOnOffBlock, TDataBlock>>;

  // The maximum number of blocks.
  static constexpr size_t nBlocksMax = 1024; // TODO: Maybe still too small, who knows.

  // Number of cells per dim of each block.
  // Eg: dim: 1    1 << dim: 2    512 / (1 << dim): 256    #cells per block: 256
  //          2              4                      128                      16384
  //          3              8                      64                       262144
  //          4              16                     32                       1048576
  static constexpr int nCellsPerBlockDim = 512 / (1 << dim);
  static_assert(nCellsPerBlockDim > 0, "The given dimension is too large");

  TBlocks blocks_;

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 BlockCoord2BlockIdx(const TCoord &blockCoord) {
    // Compute the block index.
    uint64 idx = TCode::Encode(blockCoord.template cast<Vec<uint64, dim>>());

    // Encode the quadrant to the highest bits of the index.
    std::bitset<64> quadrant;
    ForEach<dim>([&]<auto id>() {
      int &axis = blockCoord[id];

      if (axis < 0) {
        axis = -axis;
        quadrant.set(id, true);
      }
    });

    ARIA_ASSERT((idx & (std::bitset<64>().flip() << (64 - dim))) == 0,
                "The given block coord excesses the representation of the encoder, "
                "please use a larger encoder instead");
    idx |= quadrant << (64 - dim);

    return idx;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static TCoord CellCoord2BlockCoord(const TCoord &cellCoord) {
    return cellCoord / nCellsPerBlockDim;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 CellCoord2BlockIdx(const TCoord &cellCoord) {
    return BlockCoord2BlockIdx(CellCoord2BlockCoord(cellCoord));
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
