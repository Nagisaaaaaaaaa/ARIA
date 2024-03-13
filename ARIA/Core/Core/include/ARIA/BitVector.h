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
private:
  ARIA_PROP(private, private, , bool, operatorBrackets, size_t);

public:
  auto operator[](const size_t &i) const { return operatorBrackets(i); }

  auto operator[](const size_t &i) { return operatorBrackets(i); }

private:
  using TBlock = uint;
  static constexpr uint nBitsPerBlock = sizeof(TBlock) * 8;

  std::vector<TBlock> blocks_;

  static bool GetBit(const TBlock &block, uint iBits) {
    ARIA_ASSERT(iBits < nBitsPerBlock, "The given bit index should be smaller than the number of bits per block");
    return static_cast<bool>((block >> iBits) & TBlock{1});
  }

  static TBlock &FlipBit(TBlock &block, uint iBits) {
    ARIA_ASSERT(iBits < nBitsPerBlock, "The given bit index should be smaller than the number of bits per block");
    return block ^= (TBlock{1} << iBits);
  }

  static TBlock &SetBit(TBlock &block, uint iBits, bool bit) {
    ARIA_ASSERT(iBits < nBitsPerBlock, "The given bit index should be smaller than the number of bits per block");
    TBlock mask = TBlock{1} << iBits;
    return block = (block & ~mask) | (TBlock{bit} << iBits);
  }

  [[nodiscard]] bool ARIA_PROP_IMPL(operatorBrackets)(const size_t &i) const {
    size_t iBlocks = i / nBitsPerBlock;
    size_t iBits = i % nBitsPerBlock;
    return GetBit(blocks_[iBlocks], iBits);
  }

  void ARIA_PROP_IMPL(operatorBrackets)(const size_t &i, const bool &value) {
    size_t iBlocks = i / nBitsPerBlock;
    size_t iBits = i % nBitsPerBlock;
    SetBit(blocks_[iBlocks], iBits, value);
  }
};

} // namespace ARIA
