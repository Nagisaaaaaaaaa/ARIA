#pragma once

/// \file
/// \warning `BitVector` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/Property.h"

#include <cuda/atomic>

namespace ARIA {

template <typename TDerived, typename TThreadUnsafeOrSafe>
class BitVectorBase {
private:
  [[nodiscard]] const TDerived &derived() const {
    const TDerived &d = *static_cast<const TDerived *>(this);
    static_assert(std::is_same_v<decltype(d.storage()[0]), const TBlock &>,
                  "Element type of the storage should be the same as `TBlock`");
    return d;
  }

  [[nodiscard]] TDerived &derived() {
    TDerived &d = *static_cast<TDerived *>(this);
    static_assert(std::is_same_v<decltype(d.storage()[0]), TBlock &>,
                  "Element type of the storage should be the same as `TBlock`");
    return d;
  }

public:
  BitVectorBase() = default;

  ARIA_COPY_ABILITY(BitVectorBase, default);

  friend void swap(BitVectorBase &lhs, BitVectorBase &rhs) ARIA_NOEXCEPT {
    using std::swap;
    swap(lhs.nBits_, rhs.nBits_);
  }

  BitVectorBase(BitVectorBase &&other) ARIA_NOEXCEPT : BitVectorBase() { swap(*this, other); }

  BitVectorBase &operator=(BitVectorBase &&other) ARIA_NOEXCEPT {
    swap(*this, other);
    return *this;
  }

public:
  ARIA_PROP(public, public, , bool, at, size_t);

public:
  [[nodiscard]] auto operator[](size_t i) const { return at(i); }

  [[nodiscard]] auto operator[](size_t i) { return at(i); }

  TDerived &Fill(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    FillBit(derived().storage()[iBlocks], iBits);
    return derived();
  }

  TDerived &Clear(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    ClearBit(derived().storage()[iBlocks], iBits);
    return derived();
  }

  TDerived &Flip(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    FlipBit(derived().storage()[iBlocks], iBits);
    return derived();
  }

  [[nodiscard]] size_t size() const { return nBits_; }

  void resize(size_t n) {
    derived().storage().resize((n + nBitsPerBlock - 1) / nBitsPerBlock);
    nBits_ = n;
  }

protected:
  using TBlock = uint;
  static constexpr uint nBitsPerBlock = sizeof(TBlock) * 8;

  size_t nBits_ = 0;

  [[nodiscard]] std::pair<size_t, uint> i2iBlocksAndiBits(size_t i) const {
    ARIA_ASSERT(i < nBits_, "The given bit index should be smaller than the total number of bits");

    size_t iBlocks = i / nBitsPerBlock;
    uint iBits = i % nBitsPerBlock;
    return {iBlocks, iBits};
  }

  [[nodiscard]] static bool GetBit(const TBlock &block, uint iBits) {
    return static_cast<bool>((block >> iBits) & TBlock{1});
  }

  static TBlock &FillBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadUnsafeOrSafe, ThreadUnsafe>) {
      return block |= (TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadUnsafeOrSafe, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_or(TBlock{1} << iBits);
      return block;
    }
  }

  static TBlock &ClearBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadUnsafeOrSafe, ThreadUnsafe>) {
      return block &= ~(TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadUnsafeOrSafe, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_and(~(TBlock{1} << iBits));
      return block;
    }
  }

  static TBlock &FlipBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadUnsafeOrSafe, ThreadUnsafe>) {
      return block ^= (TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadUnsafeOrSafe, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_xor(TBlock{1} << iBits);
      return block;
    }
  }

  static TBlock &SetBit(TBlock &block, uint iBits, bool bit) {
    TBlock mask = TBlock{1} << iBits;
    return block = (block & ~mask) | (TBlock{bit} << iBits);
  }

  [[nodiscard]] bool ARIA_PROP_IMPL(at)(size_t i) const {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    return GetBit(derived().storage()[iBlocks], iBits);
  }

  void ARIA_PROP_IMPL(at)(size_t i, bool value) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    SetBit(derived().storage()[iBlocks], iBits, value);
  }
};

//
//
//
template <typename TSpace, typename TThreadUnsafeOrSafe>
class BitVector;

template <>
class BitVector<SpaceHost, ThreadUnsafe> : public std::vector<bool> {};

template <>
class BitVector<SpaceHost, ThreadSafe> : public BitVectorBase<BitVector<SpaceHost, ThreadSafe>, ThreadSafe> {
public:
  using Base = BitVectorBase<BitVector<SpaceHost, ThreadSafe>, ThreadSafe>;

  explicit BitVector(size_t n = 0) { resize(n); }

  ARIA_COPY_MOVE_ABILITY(BitVector, default, default);

private:
  friend Base;

  std::vector<Base::TBlock> blocks_;

  [[nodiscard]] const auto &storage() const { return blocks_; }

  [[nodiscard]] auto &storage() { return blocks_; }
};

} // namespace ARIA
