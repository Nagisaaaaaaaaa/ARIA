#pragma once

/// \file
/// \warning `BitVector` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/Property.h"

#include <bitset>
#include <vector>

namespace ARIA {

template <typename TSpace, typename TThreadUnsafeOrSafe>
class BitVector;

template <>
class BitVector<SpaceHost, ThreadUnsafe> : public std::vector<bool> {};

template <>
class BitVector<SpaceHost, ThreadSafe> {
public:
  explicit BitVector(size_t n = 0) { resize(n); }

  ARIA_COPY_ABILITY(BitVector, default);

  friend void swap(BitVector &lhs, BitVector &rhs) ARIA_NOEXCEPT {
    using std::swap;
    swap(lhs.blocks_, rhs.blocks_);
    swap(lhs.nBits_, rhs.nBits_);
  }

  BitVector(BitVector &&other) ARIA_NOEXCEPT : BitVector() { swap(*this, other); }

  BitVector &operator=(BitVector &&other) ARIA_NOEXCEPT {
    swap(*this, other);
    return *this;
  }

public:
  ARIA_PROP(public, public, , bool, at, size_t);

public:
  auto operator[](size_t i) const { return at(i); }

  auto operator[](size_t i) { return at(i); }

  size_t size() const { return nBits_; }

  void resize(size_t n) {
    blocks_.resize((n + nBitsPerBlock - 1) / nBitsPerBlock);
    nBits_ = n;
  }

private:
  using TBlock = uint;
  static constexpr uint nBitsPerBlock = sizeof(TBlock) * 8;

  std::vector<TBlock> blocks_;
  size_t nBits_ = 0;

  static std::pair<size_t, uint> i2iBlocksAndiBits(size_t i) {
    size_t iBlocks = i / nBitsPerBlock;
    uint iBits = i % nBitsPerBlock;
    return {iBlocks, iBits};
  }

  static bool GetBit(const TBlock &block, uint iBits) {
    ARIA_ASSERT(iBits < nBitsPerBlock, "The given `iBits` should be smaller than the number of bits per block");
    return static_cast<bool>((block >> iBits) & TBlock{1});
  }

  static TBlock &FlipBit(TBlock &block, uint iBits) {
    ARIA_ASSERT(iBits < nBitsPerBlock, "The given `iBits` should be smaller than the number of bits per block");
    return block ^= (TBlock{1} << iBits);
  }

  static TBlock &SetBit(TBlock &block, uint iBits, bool bit) {
    ARIA_ASSERT(iBits < nBitsPerBlock, "The given `iBits` should be smaller than the number of bits per block");
    TBlock mask = TBlock{1} << iBits;
    return block = (block & ~mask) | (TBlock{bit} << iBits);
  }

  [[nodiscard]] bool ARIA_PROP_IMPL(at)(size_t i) const {
    ARIA_ASSERT(i < nBits_, "The given bit index should be smaller than the total number of bits");
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    return GetBit(blocks_[iBlocks], iBits);
  }

  void ARIA_PROP_IMPL(at)(size_t i, bool value) {
    ARIA_ASSERT(i < nBits_, "The given bit index should be smaller than the total number of bits");
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    SetBit(blocks_[iBlocks], iBits, value);
  }
};

} // namespace ARIA
