#pragma once

/// \file
/// \warning `VDB` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/BitArray.h"
#include "ARIA/Launcher.h"
#include "ARIA/MortonCode.h"
#include "ARIA/Vec.h"

#include <stdgpu/unordered_map.cuh>
#include <thrust/host_vector.h>

namespace ARIA {

namespace vdb::detail {

// Compute $x^N$ at compile time.
template <int N, typename T>
ARIA_HOST_DEVICE static T consteval powN(T x) {
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
//
//
// A VDB handle is a lowest-level C-like VDB resource manager, which
// implements all things needed by C++ VDB classes.
// The relationship between `VDBHandle` and `VDB` is similar to that of
// `std::coroutine_handle` and `cppcoro::Task`.
//
// Introducing a handle class is necessary because
// we not only need to implement an owning `VDB`, but also need to
// implement non-owning `VDBAccessor`s.
// It is OK to implement them with CRTP, like `BitVector` and `TensorVector`, but
// such an implementation will be extremely complex for this case.
// Thus, a more C-like handle-based design is used here.
template <typename T, auto dim, typename TSpace>
class VDBHandle;

// Device VDB handle.
template <typename T, auto dim_>
class VDBHandle<T, dim_, SpaceDevice> {
public:
  using value_type = T;
  static constexpr auto dim = dim_;
  using TSpace = SpaceDevice;

public:
  // `VDBHandle` is something like (actually not) a pointer.
  // It is constructed to something like `nullptr` by default.
  VDBHandle() = default;

  // Like pointers, copy and move are trivial.
  ARIA_COPY_MOVE_ABILITY(VDBHandle, default, default);

  // Create a `VDBHandle` which points to some resource.
  [[nodiscard]] static VDBHandle Create() {
    VDBHandle handle;
    handle.blocks_ = TBlocks::createDeviceObject(nBlocksMax);
    return handle;
  }

  // Destroy the resource which the `VDBHandle` points to.
  void Destroy() noexcept /*! Actually, exceptions may be thrown here. */ {
    Launcher(1, [range = blocks_.device_range()] ARIA_DEVICE(size_t i) {
      // `block.first`  : the block index, defined by `TCode`.
      // `block.second` : the block storage, whose type is `TBlockStorage`.
      for (auto &block : range) {
        //! Device memory is dynamically allocated with `new`, so,
        //! `delete` should be called to free the memory.
        delete block.second.p;
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    TBlocks::destroyDeviceObject(blocks_);
  }

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
private:
  // See the comments below.
  template <uint n, typename... Values>
  static decltype(auto) MakeBlockLayoutImpl(Values &&...values) {
    if constexpr (n == 0)
      return make_layout_major(std::forward<Values>(values)...);
    else
      return MakeBlockLayoutImpl<n - 1>(C<nCellsPerBlockDim>{}, std::forward<Values>(values)...);
  }

  // A function wrapper which calls `make_layout_major` with `n` `nCellsPerBlockDim`s, that is,
  // `make_layout_major(C<nCellsPerBlockDim>{}, C<nCellsPerBlockDim>{}, ..., C<nCellsPerBlockDim>{})`.
  template <uint n>
  static decltype(auto) MakeBlockLayout() {
    return MakeBlockLayoutImpl<n>();
  }

  //
  //
  //
public:
  // Type of the coordinate.
  using TVec = Vec<int, dim>;

  // Type of the space filling curve encoder and decoder, which
  // is used to hash the block coord to and from the block index.
  using TCode = MortonCode<dim>;

  // Type of the layout of each block.
  using TBlockLayout = decltype(MakeBlockLayout<dim>());
  static_assert(is_static_v<TBlockLayout>, "The block layout should be a static layout");

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
      while (barrier.load(cuda::std::memory_order_acquire)) {
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

private:
  template <typename... Ts>
  friend class Launcher;

  TBlocks blocks_;

  //
  //
  //
private:
  [[nodiscard]] ARIA_HOST_DEVICE static uint64 BlockCoord2BlockIdx(const TVec &blockCoord) {
    // Compute the quadrant bits.
    uint64 quadrantBits = 0;
    ForEach<dim>([&]<auto id>() {
      int &axis = blockCoord[id];

      if (axis < 0) {
        axis = -axis;
        quadrantBits |= (1 << id); // Fill the `id`^th bit.
      }
    });

    // Compute the block index.
    uint64 idx = TCode::Encode(Auto(blockCoord.template cast<uint64>()));

    // Encode the quadrant to the highest bits of the index.
    ARIA_ASSERT((idx & ((~uint64{0}) << (64 - dim))) == 0,
                "The given block coord excesses the representation of the encoder, "
                "please use a larger encoder instead");
    idx |= quadrantBits << (64 - dim);

    return idx;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static TVec BlockIdx2BlockCoord(const uint64 &blockIdx) {
    TVec quadrant;
    quadrant.fill(1);

    // Compute the quadrant.
    uint64 quadrantBits = blockIdx >> (64 - dim);

    ForEach<dim>([&]<auto id>() {
      if (quadrantBits & (1 << id)) {
        quadrant[id] = -quadrant[id];
      }
    });

    // Compute the block coord.
    uint64 idx = blockIdx & ((~uint64{0}) >> dim);
    return TCode::Decode(idx).template cast<typename TVec::Scalar>().cwiseProduct(quadrant);
  }

  [[nodiscard]] ARIA_HOST_DEVICE static TVec CellCoord2BlockCoord(const TVec &cellCoord) {
    // TODO: Compiler bug here: `nCellsPerBlockDim` is not defined in device code.
    // return cellCoord / nCellsPerBlockDim;

    constexpr auto n = nCellsPerBlockDim;
    return cellCoord / n;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static TVec BlockCoord2CellCoordOffset(const TVec &blockCoord) {
    // TODO: Compiler bug here: `nCellsPerBlockDim` is not defined in device code.
    // return cellCoord * nCellsPerBlockDim;

    constexpr auto n = nCellsPerBlockDim;
    return blockCoord * n;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 CellCoord2BlockIdx(const TVec &cellCoord) {
    return BlockCoord2BlockIdx(CellCoord2BlockCoord(cellCoord));
  }

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 CellCoord2CellIdxInBlock(const TVec &cellCoord) {
    TVec cellCoordInBlock;
    ForEach<dim>([&]<auto d>() { cellCoordInBlock[d] = cellCoord[d] % nCellsPerBlockDim; });
    return TBlockLayout{}(ToCoord(cellCoordInBlock));
  }

private:
  ARIA_HOST_DEVICE const TBlock &block_AssumeExist(const TVec &cellCoord) const {
#if ARIA_IS_DEVICE_CODE
    // Get reference to the already emplaced block.
    return blocks_.find(CellCoord2BlockIdx(cellCoord))->second;
#else
    ARIA_STATIC_ASSERT_FALSE("This method should not be called at host side");
#endif
  }

  ARIA_HOST_DEVICE TBlock &block_AllocateIfNotExist(const TVec &cellCoord) {
#if ARIA_IS_DEVICE_CODE
    // Each thread is trying to insert an empty block into the unordered map,
    // but only one unique thread will succeed.
    auto res = blocks_.emplace(CellCoord2BlockIdx(cellCoord), TBlock{});

    // If success, get pointer to the emplaced block.
    TBlock *block = &res.first->second;

    if (res.second) { // For the unique thread which succeeded to emplace the block:
      // Allocate the block storage.
      block->p = new TBlockStorage();

      // Mark the storage as ready.
      block->arrive();
    } else { // For other threads which failed:
      // Get reference to the emplaced block.
      block = &blocks_.find(CellCoord2BlockIdx(cellCoord))->second;

      // Wait for the storage being ready.
      block->wait();
    }

    // For now, all threads have access to the emplaced block.
    return *block;
#else
    ARIA_STATIC_ASSERT_FALSE("This method should not be called at host side");
#endif
  }

public:
  ARIA_PROP(public, public, ARIA_HOST_DEVICE, T, value_AssumeExist, TVec);

  ARIA_PROP(public, public, ARIA_HOST_DEVICE, T, value_AllocateIfNotExist, TVec);

private:
  [[nodiscard]] ARIA_HOST_DEVICE T ARIA_PROP_IMPL(value_AssumeExist)(const TVec &cellCoord) const {
#if ARIA_IS_DEVICE_CODE
    const TBlock &b = block_AssumeExist(cellCoord);
    ARIA_ASSERT(b.p->onOff[CellCoord2CellIdxInBlock(cellCoord)]);
    return b.p->data[CellCoord2CellIdxInBlock(cellCoord)];
#else
    ARIA_STATIC_ASSERT_FALSE("This method should not be called at host side");
#endif
  }

  ARIA_HOST_DEVICE void ARIA_PROP_IMPL(value_AssumeExist)(const TVec &cellCoord, const T &value) {
#if ARIA_IS_DEVICE_CODE
    const TBlock &b = block_AssumeExist(cellCoord);
    b.p->onOff.Fill(CellCoord2CellIdxInBlock(cellCoord));
    b.p->data[CellCoord2CellIdxInBlock(cellCoord)] = value;
#else
    ARIA_STATIC_ASSERT_FALSE("This method should not be called at host side");
#endif
  }

  [[nodiscard]] ARIA_HOST_DEVICE T ARIA_PROP_IMPL(value_AllocateIfNotExist)(const TVec &cellCoord) const {
    return ARIA_PROP_IMPL(value_AssumeExist)(cellCoord);
  }

  ARIA_HOST_DEVICE void ARIA_PROP_IMPL(value_AllocateIfNotExist)(const TVec &cellCoord, const T &value) {
#if ARIA_IS_DEVICE_CODE
    TBlock &b = block_AllocateIfNotExist(cellCoord);
    b.p->onOff.Fill(CellCoord2CellIdxInBlock(cellCoord));
    b.p->data[CellCoord2CellIdxInBlock(cellCoord)] = value;
#else
    ARIA_STATIC_ASSERT_FALSE("This method should not be called at host side");
#endif
  }
};

//
//
//
template <typename T, auto dim, typename TSpace>
class VDB;

//
//
//
struct AllocateWrite {};

struct Write {};

struct Read {};

template <typename T, auto dim, typename TSpace, typename TAccessor>
class VDBAccessor;

// Write or allocate-write accessor.
template <typename T, auto dim, typename TSpace, typename TAccessor>
  requires(std::is_same_v<TAccessor, Write> || std::is_same_v<TAccessor, AllocateWrite>)
class VDBAccessor<T, dim, TSpace, TAccessor> {
private:
  using THandle = VDBHandle<T, dim, TSpace>;
  using TVec = THandle::TVec;

  static constexpr bool allocateIfNotExist = std::is_same_v<TAccessor, AllocateWrite>;

public:
  VDBAccessor() = default;

  ARIA_COPY_MOVE_ABILITY(VDBAccessor, default, default);

private:
  friend class VDB<T, dim, TSpace>;

  ARIA_HOST_DEVICE explicit VDBAccessor(THandle handle) : handle_(std::move(handle)) {}

public:
  [[nodiscard]] ARIA_HOST_DEVICE decltype(auto) value(const TVec &cellCoord) {
    if constexpr (allocateIfNotExist)
      return handle_.value_AllocateIfNotExist(cellCoord);
    else
      return handle_.value_AssumeExist(cellCoord);
  }

  [[nodiscard]] ARIA_HOST_DEVICE decltype(auto) value(const TVec &cellCoord) const {
    if constexpr (allocateIfNotExist)
      return handle_.value_AllocateIfNotExist(cellCoord);
    else
      return handle_.value_AssumeExist(cellCoord);
  }

private:
  THandle handle_;
};

// Read accessor.
template <typename T, auto dim, typename TSpace, typename TAccessor>
  requires(std::is_same_v<TAccessor, Read>)
class VDBAccessor<T, dim, TSpace, TAccessor> {
private:
  using THandle = VDBHandle<T, dim, TSpace>;
  using TVec = THandle::TVec;

public:
  VDBAccessor() = default;

  ARIA_COPY_MOVE_ABILITY(VDBAccessor, default, default);

private:
  friend class VDB<T, dim, TSpace>;

  ARIA_HOST_DEVICE explicit VDBAccessor(THandle handle) : handle_(std::move(handle)) {}

public:
  [[nodiscard]] ARIA_HOST_DEVICE decltype(auto) value(const TVec &cellCoord) const {
    return handle_.value_AssumeExist(cellCoord);
  }

private:
  THandle handle_;
};

//
//
//
template <typename T, auto dim, typename TSpace>
class VDB {
public:
  VDB() : handle_(std::make_unique<THandle>(THandle::Create())) {}

  ARIA_COPY_MOVE_ABILITY(VDB, delete, default);

  ~VDB() noexcept {
    if (handle_)
      handle_->Destroy();
  }

public:
  using value_type = T;

  using AllocateWriteAccessor = VDBAccessor<T, dim, TSpace, AllocateWrite>;
  using WriteAccessor = VDBAccessor<T, dim, TSpace, Write>;
  using ReadAccessor = VDBAccessor<T, dim, TSpace, Read>;

public:
  [[nodiscard]] AllocateWriteAccessor allocateWriteAccessor() { return AllocateWriteAccessor{*handle_}; }

  [[nodiscard]] WriteAccessor writeAccessor() { return WriteAccessor{*handle_}; }

  [[nodiscard]] ReadAccessor readAccessor() const { return ReadAccessor{*handle_}; }

private:
  using THandle = VDBHandle<T, dim, TSpace>;
  using TVec = THandle::TVec;

  std::unique_ptr<THandle> handle_;

  template <typename... Ts>
  friend class Launcher;
};

//
//
//
template <typename T>
struct is_vdb : std::false_type {};

template <typename T, auto dim, typename TSpace>
struct is_vdb<VDB<T, dim, TSpace>> : std::true_type {};

template <typename T>
static constexpr bool is_vdb_v = is_vdb<T>::value;

template <typename T>
concept VDBType = is_vdb_v<T>;

//
template <typename T>
struct is_host_vdb : std::false_type {};

template <typename T, auto dim>
struct is_host_vdb<VDB<T, dim, SpaceHost>> : std::true_type {};

template <typename T>
static constexpr bool is_host_vdb_v = is_host_vdb<T>::value;

template <typename T>
concept HostVDBType = is_host_vdb_v<T>;

//
template <typename T>
struct is_device_vdb : std::false_type {};

template <typename T, auto dim>
struct is_device_vdb<VDB<T, dim, SpaceDevice>> : std::true_type {};

template <typename T>
static constexpr bool is_device_vdb_v = is_device_vdb<T>::value;

template <typename T>
concept DeviceVDBType = is_device_vdb_v<T>;

//
//
//
template <typename VDB>
using VDBAllocateWriteAccessor = typename VDB::AllocateWriteAccessor;

template <typename VDB>
using VDBWriteAccessor = typename VDB::WriteAccessor;

template <typename VDB>
using VDBReadAccessor = typename VDB::ReadAccessor;

//
//
//
// Launcher.
template <typename THandle, typename F>
ARIA_KERNEL static void
KernelLaunchVDBBlock(typename THandle::TBlock block, decltype(ToCoord(typename THandle::TVec{})) cellCoordOffset, F f) {
  using TVec = typename THandle::TVec;
  using TBlockLayout = typename THandle::TBlockLayout;
  static constexpr auto dim = THandle::dim;

  int i = static_cast<int>(threadIdx.x) + static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
  if (i >= cosize_safe_v<TBlockLayout>)
    return;

  auto cellCoordInBlock = Auto(TBlockLayout{}.get_hier_coord(i));
  TVec cellCoord = ToVec(cellCoordOffset + cellCoordInBlock);

  if (block.p->onOff[i])
    f(cellCoord);
}

template <DeviceVDBType TVDB, typename F>
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
    auto range = handle_.blocks_.device_range();
    thrust::host_vector<stdgpu::pair<uint64, TBlock>> rangeH(range.size());
    thrust::copy(range.begin(), range.end(), rangeH.begin());

    for (auto &block : rangeH) {
      // Compute block coord from block idx.
      TVec blockCoord = handle_.BlockIdx2BlockCoord(block.first);
      TVec cellCoordOffset = handle_.BlockCoord2CellCoordOffset(blockCoord);

      // Launch.
      Base::Launch(KernelLaunchVDBBlock<THandle, F>, block.second, ToCoord(cellCoordOffset), f_);
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

template <DeviceVDBType TVDB, typename F>
Launcher(const TVDB &vdb, const F &f) -> Launcher<TVDB, F>;

} // namespace ARIA
