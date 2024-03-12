#pragma once

/// \file
/// \warning `VDB` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/MortonCode.h"
#include "ARIA/TensorVector.h"
#include "ARIA/Vec.h"

#include <stdgpu/unordered_set.cuh>

#include <bitset>

namespace ARIA {

template <typename T, auto dim, typename TSpace>
class VDB;

//
//
//
// Device VDB.
template <typename T, auto dim>
class VDB<T, dim, SpaceDevice> {
public:
  VDB() : blockIndicesToAllocate_(stdgpu::unordered_set<uint64>::createDeviceObject(toAllocateCapacity)) {}

  ARIA_COPY_MOVE_ABILITY(VDB, delete, default);

  ~VDB() noexcept /*Actually, exceptions may be thrown here.*/ {
    stdgpu::unordered_set<uint64>::destroyDeviceObject(blockIndicesToAllocate_);
  }

public:
  // TODO: Implement IsValueOn().
  // TODO: Implement GetValue().

private:
  // Type of the coordinate.
  using TCoord = Vec<int, dim>;

  // The space filling curve encoder and decoder used to hash the block coord to and from the block index.
  using Code = MortonCode<dim>;

  static constexpr size_t toAllocateCapacity = 1024; // TODO: Maybe still too small, who knows.

  // Eg: dim: 1    1 << dim: 2    512 / (1 << dim): 256    #cells per block: 256
  //          2              4                      128                      16384
  //          3              8                      64                       262144
  //          4              16                     32                       1048576
  static constexpr int blockDim = 512 / (1 << dim);
  static_assert(blockDim > 0, "The given dimension is too large");

  stdgpu::unordered_set<uint64> blockIndicesToAllocate_;
  thrust::device_vector<uint64> blockIndicesAllocated_;

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 BlockCoord2BlockIdx(const TCoord &blockCoord) {
    // Compute the block index.
    uint64 idx = Code::Encode(blockCoord.template cast<Vec<uint64, dim>>());

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
    return cellCoord / blockDim;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 CellCoord2BlockIdx(const TCoord &cellCoord) {
    return BlockCoord2BlockIdx(CellCoord2BlockCoord(cellCoord));
  }

  ARIA_DEVICE void MarkBlockOnByBlockCoord(const TCoord &blockCoord) {
    blockIndicesToAllocate_.insert(BlockCoord2BlockIdx(blockCoord));
  }

  ARIA_DEVICE void MarkBlockOnByCellCoord(const TCoord &cellCoord) {
    blockIndicesToAllocate_.insert(CellCoord2BlockIdx(cellCoord));
  }

  void AllocateBlocks() {
    thrust::host_vector<uint64> indicesH(blockIndicesToAllocate_.size());
    thrust::copy(blockIndicesToAllocate_.device_range().begin(), blockIndicesToAllocate_.device_range().end(),
                 indicesH.begin());

    // TODO: Allocate all the corresponding blocks.
  }
};

} // namespace ARIA
