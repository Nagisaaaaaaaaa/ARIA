#pragma once

#include "ARIA/BitArray.h"
#include "ARIA/Launcher.h"
#include "ARIA/MortonCode.h"
#include "ARIA/Vec.h"

#include <stdgpu/unordered_map.cuh>
#include <thrust/device_vector.h>
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

//
//
//
//
//
// Fwd.
template <typename T, auto dim, typename TSpace>
class VDB;

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
      // `block.first`  : Code of the block coord, defined by `TCode`.
      // `block.second` : The block, whose type is `TBlock`.
      for (auto &block : range) {
        //! Device memory is dynamically allocated with `new`, so,
        //! `delete` should be called to free the memory.
        delete block.second.storage();
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
  // Type of the coordinate, represented with `Vec`.
  using TVec = Vec<int, dim>;

  // Type of the coordinate, represented with `Coord`.
  using TCoord = decltype(ToCoord(TVec{}));

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
  //
  //! When many threads simultaneously access an unallocated block,
  //! only one thread will get the right to allocate the block storage.
  //! Every thread knows whether it is the winner, and at the same time,
  //! a barrier is constructed to "ON".
  //!
  //! For the winner thread, it allocates and initialize the block storage.
  //! After that, it calls `arrive` to mark the storage as ready.
  //!
  //! For other threads, they calls `wait` to wait for the storage being ready.
  //! After `arrive` is called by the winner, they are able to access the storage.
  class TBlock {
  public:
    // The block storage pointer.
    ARIA_REF_PROP(public, ARIA_HOST_DEVICE, storage, storage_);

  public:
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
    TBlockStorage *storage_ = nullptr;
    uint barrier_ = 1; // Constructed to "ON".
  };

  // Type of the sparse blocks tree, which is a device unordered map:
  //   Key  : Code of the block coord, defined by `TCode`.
  //   Value: The block.
  using TBlocks = stdgpu::unordered_map<uint64, TBlock>;

private:
  // Something like a pointer.
  TBlocks blocks_;

  //
  //
  //
  // Conversion between different coordinate systems.
private:
  [[nodiscard]] ARIA_HOST_DEVICE static uint64 BlockCoord2BlockIdx(TVec blockCoord) {
    // Compute the quadrant bits and remove the signs.
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
                "please contact the developers and use a larger encoder instead");
    idx |= quadrantBits << (64 - dim);

    return idx;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static TVec BlockIdx2BlockCoord(const uint64 &blockIdx) {
    TVec quadrant;
    quadrant.fill(1);

    // Compute the quadrant.
    uint64 quadrantBits = blockIdx >> (64 - dim);

    ForEach<dim>([&]<auto id>() {
      if (quadrantBits & (1 << id)) { // Check the `id`^th bit.
        quadrant[id] = -quadrant[id];
      }
    });

    // Decode the quadrant from the highest bits of the index.
    uint64 idx = blockIdx & ((~uint64{0}) >> dim);

    // Compute the block coord and add the signs.
    TVec blockCoord = TCode::Decode(idx).template cast<typename TVec::Scalar>();
    return blockCoord.cwiseProduct(quadrant);
  }

  [[nodiscard]] ARIA_HOST_DEVICE static TVec CellCoord2BlockCoord(const TVec &cellCoord) {
    // TODO: Compiler bug here: `nCellsPerBlockDim` is not defined in device code.
    constexpr auto n = nCellsPerBlockDim;

    TVec blockCoord;
    ForEach<dim>(
        [&]<auto id>() { blockCoord[id] = cellCoord[id] >= 0 ? (cellCoord[id] / n) : ((cellCoord[id] - n + 1) / n); });
    return blockCoord;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static TVec CellCoord2CellCoordInBlock(const TVec &cellCoord) {
    constexpr auto n = nCellsPerBlockDim;
    return cellCoord - CellCoord2BlockCoord(cellCoord) * n;
  }

  // `cellCoord` = `CellCoordOffset` + `cellCoordInBlock`.
  [[nodiscard]] ARIA_HOST_DEVICE static TVec BlockCoord2CellCoordOffset(const TVec &blockCoord) {
    constexpr auto n = nCellsPerBlockDim;
    return blockCoord * n;
  }

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 CellCoord2BlockIdx(const TVec &cellCoord) {
    return BlockCoord2BlockIdx(CellCoord2BlockCoord(cellCoord));
  }

  [[nodiscard]] ARIA_HOST_DEVICE static uint64 CellCoord2CellIdxInBlock(const TVec &cellCoord) {
    return TBlockLayout{}(ToCoord(CellCoord2CellCoordInBlock(cellCoord)));
  }

  //
  //
  //
private:
  // Get reference to the block, which the `cellCoord` is located in.
  // Assume that the block may not exist, so,
  // dynamic memory allocation may be performed here.
  ARIA_HOST_DEVICE TBlock &block_AllocateIfNotExist(const TVec &cellCoord) {
#if ARIA_IS_DEVICE_CODE
    // Each thread is trying to insert a block with zero storage into the unordered map,
    // but only one unique thread will succeed.
    auto res = Auto(blocks_.emplace(CellCoord2BlockIdx(cellCoord), TBlock{}));
    TBlock *block = &res.first->second;

    if (res.second) { // For the unique thread which succeeded in emplacing the block, `block` points to that block.
      // Allocate the block storage.
      block->storage() = new TBlockStorage();

      // Mark the storage as ready.
      block->arrive();
    } else { // For other threads which failed to emplace the block, `block` points to an undefined memory.
      // Get reference to the emplaced block.
      block = &blocks_.find(CellCoord2BlockIdx(cellCoord))->second;

      // Wait for the storage being ready.
      block->wait();
    }

    // For now, all threads have access to the emplaced block.
    return *block;
#else
    ARIA_STATIC_ASSERT_FALSE("This method is not allowed to be called at host side");
#endif
  }

  // Assume that the block is already exist.
  ARIA_HOST_DEVICE const TBlock &block_AssumeExist(const TVec &cellCoord) const {
#if ARIA_IS_DEVICE_CODE
    // Get reference to the already emplaced block.
    return blocks_.find(CellCoord2BlockIdx(cellCoord))->second;
#else
    ARIA_STATIC_ASSERT_FALSE("This method is not allowed to be called at host side");
#endif
  }

  // Get the block only if it exists, else, return `nullptr`.
  ARIA_HOST_DEVICE const TBlock *block_GetIfExist(const TVec &cellCoord) const {
#if ARIA_IS_DEVICE_CODE
    auto it = Auto(blocks_.find(CellCoord2BlockIdx(cellCoord)));
    if (it == blocks_.end())
      return nullptr;

    return &it->second;
#else
    ARIA_STATIC_ASSERT_FALSE("This method is not allowed to be called at host side");
#endif
  }

  ARIA_HOST_DEVICE TBlock *block_GetIfExist(const TVec &cellCoord) {
#if ARIA_IS_DEVICE_CODE
    auto it = Auto(blocks_.find(CellCoord2BlockIdx(cellCoord)));
    if (it == blocks_.end())
      return nullptr;

    return &it->second;
#else
    ARIA_STATIC_ASSERT_FALSE("This method is not allowed to be called at host side");
#endif
  }

  //
  //
  //
public:
  // Get or set the value at `cellCoord`.
  ARIA_PROP(public, public, ARIA_HOST_DEVICE, T, value_AllocateIfNotExist, TVec);

  ARIA_PROP(public, public, ARIA_HOST_DEVICE, T, value_AssumeExist, TVec);

private:
  //! It is considered undefined behavior to get the value which has not been set yet.
  //! So, the getter of `value_AllocateIfNotExist` can be implemented the same as `value_AssumeExist`'s.
  [[nodiscard]] ARIA_HOST_DEVICE T ARIA_PROP_IMPL(value_AllocateIfNotExist)(const TVec &cellCoord) const {
    return ARIA_PROP_IMPL(value_AssumeExist)(cellCoord);
  }

  ARIA_HOST_DEVICE void ARIA_PROP_IMPL(value_AllocateIfNotExist)(const TVec &cellCoord, const T &value) {
#if ARIA_IS_DEVICE_CODE
    TBlock &b = block_AllocateIfNotExist(cellCoord);

    auto cellIdxInBlock = Auto(CellCoord2CellIdxInBlock(cellCoord));
    b.storage()->onOff.Fill(cellIdxInBlock);
    b.storage()->data[cellIdxInBlock] = value;
#else
    ARIA_STATIC_ASSERT_FALSE("This method is not allowed to be called at host side");
#endif
  }

  // Set the value at `cellCoord` to off.
  ARIA_HOST_DEVICE void ARIA_PROP_IMPL(value_AllocateIfNotExist)(const TVec &cellCoord, const Off &off) {
    ARIA_PROP_IMPL(value_AssumeExist)(cellCoord, off);
  }

  [[nodiscard]] ARIA_HOST_DEVICE T ARIA_PROP_IMPL(value_AssumeExist)(const TVec &cellCoord) const {
#if ARIA_IS_DEVICE_CODE
    const TBlock &b = block_AssumeExist(cellCoord);

    auto cellIdxInBlock = Auto(CellCoord2CellIdxInBlock(cellCoord));
    ARIA_ASSERT(b.storage()->onOff[cellIdxInBlock],
                "It is considered undefined behavior to get the value which has not been set yet");
    return b.storage()->data[cellIdxInBlock];
#else
    ARIA_STATIC_ASSERT_FALSE("This method is not allowed to be called at host side");
#endif
  }

  ARIA_HOST_DEVICE void ARIA_PROP_IMPL(value_AssumeExist)(const TVec &cellCoord, const T &value) {
#if ARIA_IS_DEVICE_CODE
    const TBlock &b = block_AssumeExist(cellCoord);

    auto cellIdxInBlock = Auto(CellCoord2CellIdxInBlock(cellCoord));
    b.storage()->onOff.Fill(cellIdxInBlock);
    b.storage()->data[cellIdxInBlock] = value;
#else
    ARIA_STATIC_ASSERT_FALSE("This method is not allowed to be called at host side");
#endif
  }

  ARIA_HOST_DEVICE void ARIA_PROP_IMPL(value_AssumeExist)(const TVec &cellCoord, const Off &off) {
#if ARIA_IS_DEVICE_CODE
    // Try and get the block.
    TBlock *b = block_GetIfExist(cellCoord);

    // If the block does not exist, it is already "off", do nothing and return.
    if (!b)
      return;

    // If the block exists, clear the storage.
    auto cellIdxInBlock = Auto(CellCoord2CellIdxInBlock(cellCoord));
    b->storage()->onOff.Clear(cellIdxInBlock);
    b->storage()->data[cellIdxInBlock] = T{};
#else
    ARIA_STATIC_ASSERT_FALSE("This method is not allowed to be called at host side");
#endif
  }

  //
  //
  //
public:
  void ShrinkToFit() {
    // This variable contains whether each block should be preserved (should not be erased).
    thrust::device_vector<bool> shouldPreserveD(blocks_.max_size());

    // If there exists one `cellCoord` which is "on" within this block, mark the block as preserved.
    Launcher(*this, [=, *this, shouldPreserve = shouldPreserveD.data()] ARIA_DEVICE(const TVec &cellCoord) {
      // In `[0, max_size())`.
      size_t blockIdxInMap = blocks_.find(CellCoord2BlockIdx(cellCoord)) - blocks_.begin();
      shouldPreserve[blockIdxInMap] = true;
    }).Launch();

    // Erase all the blocks which should not be preserved.
    Launcher(blocks_.max_size(), [=, *this, shouldPreserve = shouldPreserveD.data()] ARIA_DEVICE(size_t i) mutable {
      // Return if this block should be preserved.
      if (shouldPreserve[i])
        return;

      auto block = Auto(blocks_.begin() + i);

      // Delete the storage.
      delete block->second.storage();
      // Erase from unordered map.
      blocks_.erase(block->first);
    }).Launch();

    cuda::device::current::get().synchronize();
  }

  //
  //
  //
private:
  friend class VDB<T, dim, TSpace>;

  template <typename... Ts>
  friend class ARIA::Launcher;
};

//
//
//
template <typename T>
struct is_vdb_handle : std::false_type {};

template <typename T, auto dim, typename TSpace>
struct is_vdb_handle<VDBHandle<T, dim, TSpace>> : std::true_type {};

template <typename T>
static constexpr bool is_vdb_handle_v = is_vdb_handle<T>::value;

template <typename T>
concept VDBHandleType = is_vdb_handle_v<T>;

//
template <typename T>
struct is_host_vdb_handle : std::false_type {};

template <typename T, auto dim>
struct is_host_vdb_handle<VDBHandle<T, dim, SpaceHost>> : std::true_type {};

template <typename T>
static constexpr bool is_host_vdb_handle_v = is_host_vdb_handle<T>::value;

template <typename T>
concept HostVDBHandleType = is_host_vdb_handle_v<T>;

//
template <typename T>
struct is_device_vdb_handle : std::false_type {};

template <typename T, auto dim>
struct is_device_vdb_handle<VDBHandle<T, dim, SpaceDevice>> : std::true_type {};

template <typename T>
static constexpr bool is_device_vdb_handle_v = is_device_vdb_handle<T>::value;

template <typename T>
concept DeviceVDBHandleType = is_device_vdb_handle_v<T>;

//
//
//
//
//
// Accessor policies.
struct AllocateWrite {};

struct Write {};

struct Read {};

//
//
//
// A `VDBAccessor` is a non-owning view of a `VDB`.
template <typename T, auto dim, typename TSpace, typename TAccessor>
class VDBAccessor;

// Allocate-write or write accessor.
template <typename T, auto dim, typename TSpace, typename TAccessor>
  requires(std::is_same_v<TAccessor, AllocateWrite> || std::is_same_v<TAccessor, Write>)
class VDBAccessor<T, dim, TSpace, TAccessor> {
private:
  using THandle = VDBHandle<T, dim, TSpace>;
  using TVec = THandle::TVec;

public:
  VDBAccessor() = default;

  ARIA_COPY_MOVE_ABILITY(VDBAccessor, default, default);

private:
  friend class VDB<T, dim, TSpace>;

  // This constructor should only be called by `VDB`.
  ARIA_HOST_DEVICE explicit VDBAccessor(THandle handle) : handle_(std::move(handle)) {}

public:
  /// \brief Get or set the value at `cellCoord`.
  [[nodiscard]] ARIA_HOST_DEVICE decltype(auto) value(const TVec &cellCoord) {
    if constexpr (allocateIfNotExist)
      return handle_.value_AllocateIfNotExist(cellCoord);
    else
      return handle_.value_AssumeExist(cellCoord);
  }

  /// \brief Get or set the value at `cellCoord`.
  [[nodiscard]] ARIA_HOST_DEVICE decltype(auto) value(const TVec &cellCoord) const {
    if constexpr (allocateIfNotExist)
      return handle_.value_AllocateIfNotExist(cellCoord);
    else
      return handle_.value_AssumeExist(cellCoord);
  }

private:
  THandle handle_;

  static constexpr bool allocateIfNotExist = std::is_same_v<TAccessor, AllocateWrite>;
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
  /// \brief Get or set the value at `cellCoord`.
  [[nodiscard]] ARIA_HOST_DEVICE decltype(auto) value(const TVec &cellCoord) const {
    return handle_.value_AssumeExist(cellCoord);
  }

private:
  THandle handle_;
};

//
//
//
//
//
// A `VDB` is an owning object which can generate non-owning `VDBAccessor`s.
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
  /// \brief Get the allocate-write accessor.
  [[nodiscard]] AllocateWriteAccessor allocateWriteAccessor() { return AllocateWriteAccessor{*handle_}; }

  /// \brief Get the write accessor.
  [[nodiscard]] WriteAccessor writeAccessor() { return WriteAccessor{*handle_}; }

  /// \brief Get the read accessor.
  [[nodiscard]] ReadAccessor readAccessor() const { return ReadAccessor{*handle_}; }

public:
  void ShrinkToFit() { handle_->ShrinkToFit(); }

private:
  using THandle = VDBHandle<T, dim, TSpace>;
  using TVec = THandle::TVec;

  std::unique_ptr<THandle> handle_;

private:
  template <typename... Ts>
  friend class ARIA::Launcher;
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
//
//
// Launch kernel for each cell coord with value on.
template <vdb::detail::DeviceVDBHandleType THandle, typename F>
ARIA_KERNEL static void
KernelLaunchVDBBlock(typename THandle::TBlock block, typename THandle::TCoord cellCoordOffset, F f) {
  using TVec = typename THandle::TVec;
  using TCoord = typename THandle::TCoord;
  using TBlockLayout = typename THandle::TBlockLayout;

  int cellIdxInBlock = static_cast<int>(threadIdx.x) + static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
  if (cellIdxInBlock >= cosize_safe_v<TBlockLayout>)
    return;

  TCoord cellCoordInBlock = TBlockLayout{}.get_hier_coord(cellIdxInBlock);
  TVec cellCoord = ToVec(cellCoordOffset + cellCoordInBlock);

  if (block.storage()->onOff[cellIdxInBlock])
    f(cellCoord);
}

} // namespace vdb::detail

//
//
//
template <vdb::detail::DeviceVDBHandleType THandle, typename F>
class Launcher<THandle, F> : public launcher::detail::LauncherBase<Launcher<THandle, F>> {
private:
  using Base = launcher::detail::LauncherBase<Launcher<THandle, F>>;

public:
  Launcher(const THandle &handle, const F &f) : handle_(handle), f_(f) {
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
  using TVec = THandle::TVec;
  using TBlock = THandle::TBlock;
  using TBlockLayout = THandle::TBlockLayout;

  THandle handle_;
  F f_;
};

template <vdb::detail::DeviceVDBHandleType THandle, typename F>
Launcher(const THandle &handle, const F &f) -> Launcher<THandle, F>;

} // namespace ARIA
