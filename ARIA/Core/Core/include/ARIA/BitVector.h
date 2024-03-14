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
#include <thrust/device_vector.h>

namespace ARIA {

template <typename TDerived>
class BitVectorCRTPBase {
protected:
  using TBlock = uint;
  static constexpr uint nBitsPerBlock = sizeof(TBlock) * 8;

  [[nodiscard]] const TDerived &derived() const { return *static_cast<const TDerived *>(this); }

  [[nodiscard]] TDerived &derived() { return *static_cast<TDerived *>(this); }
};

template <typename TDerived, typename TThreadSafety>
class BitVectorSpanAPI : public BitVectorCRTPBase<TDerived> {
protected:
  using Base = BitVectorCRTPBase<TDerived>;
  // clang-format off
  using typename Base::TBlock;
  using Base::nBitsPerBlock;
  using Base::derived;
  // clang-format on

public:
  BitVectorSpanAPI() = default;

  ARIA_COPY_ABILITY(BitVectorSpanAPI, default);

  friend void swap(BitVectorSpanAPI &lhs, BitVectorSpanAPI &rhs) ARIA_NOEXCEPT {
    using std::swap;
    swap(lhs.nBits_, rhs.nBits_);
  }

  BitVectorSpanAPI(BitVectorSpanAPI &&other) ARIA_NOEXCEPT : BitVectorSpanAPI() { swap(*this, other); }

  BitVectorSpanAPI &operator=(BitVectorSpanAPI &&other) ARIA_NOEXCEPT {
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
    FillBit(derived().data()[iBlocks], iBits);
    return derived();
  }

  TDerived &Clear(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    ClearBit(derived().data()[iBlocks], iBits);
    return derived();
  }

  TDerived &Flip(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    FlipBit(derived().data()[iBlocks], iBits);
    return derived();
  }

  [[nodiscard]] size_t size() const { return nBits_; }

protected:
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

  [[nodiscard]] static bool GetBit(const thrust::device_reference<const TBlock> &block, uint iBits) {
    return static_cast<bool>((block >> iBits) & TBlock{1});
  }

  static TBlock &FillBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadSafety, ThreadUnsafe>) {
      return block |= (TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadSafety, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_or(TBlock{1} << iBits);
      return block;
    }
  }

  static TBlock &ClearBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadSafety, ThreadUnsafe>) {
      return block &= ~(TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadSafety, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_and(~(TBlock{1} << iBits));
      return block;
    }
  }

  static TBlock &FlipBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadSafety, ThreadUnsafe>) {
      return block ^= (TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadSafety, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_xor(TBlock{1} << iBits);
      return block;
    }
  }

  static TBlock &SetBit(TBlock &block, uint iBits, bool bit) {
    TBlock mask = TBlock{1} << iBits;
    return block = (block & ~mask) | (TBlock{bit} << iBits);
  }

  static thrust::device_reference<TBlock> SetBit(thrust::device_reference<TBlock> block, uint iBits, bool bit) {
    TBlock mask = TBlock{1} << iBits;
    return block = (block & ~mask) | (TBlock{bit} << iBits);
  }

  [[nodiscard]] bool ARIA_PROP_IMPL(at)(size_t i) const {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    return GetBit(derived().data()[iBlocks], iBits);
  }

  void ARIA_PROP_IMPL(at)(size_t i, bool value) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    SetBit(derived().data()[iBlocks], iBits, value);
  }
};

template <typename TDerived, typename TThreadSafety>
class BitVectorStorageAPI : public BitVectorSpanAPI<TDerived, TThreadSafety> {
protected:
  using Base = BitVectorSpanAPI<TDerived, TThreadSafety>;
  // clang-format off
  using typename Base::TBlock;
  using Base::nBitsPerBlock;
  using Base::derived;

  using Base::nBits_;
  // clang-format on

public:
  void resize(size_t n) {
    derived().storage().resize((n + nBitsPerBlock - 1) / nBitsPerBlock);
    nBits_ = n;
  }
};

//
//
//
template <typename TSpace, typename TThreadSafety, typename TPtr>
class BitVectorSpan : public BitVectorSpanAPI<BitVectorSpan<TSpace, TThreadSafety, TPtr>, TThreadSafety> {
public:
  BitVectorSpan(TPtr p, size_t n) : p_(p) { nBits_ = n; }

  ARIA_COPY_MOVE_ABILITY(BitVectorSpan, default, default);

public:
  [[nodiscard]] auto data() const { return p_; }

  [[nodiscard]] auto data() { return p_; }

private:
  using Base = BitVectorSpanAPI<BitVectorSpan<TSpace, TThreadSafety, TPtr>, TThreadSafety>;
  using Base::nBits_;

  TPtr p_;
};

//
//
//
template <typename TSpace, typename TThreadSafety>
class BitVector;

template <typename TThreadSafety>
class BitVector<SpaceHost, TThreadSafety>
    : public BitVectorStorageAPI<BitVector<SpaceHost, TThreadSafety>, TThreadSafety> {
public:
  explicit BitVector(size_t n = 0) { Base::resize(n); }

  ARIA_COPY_MOVE_ABILITY(BitVector, default, default);

public:
  [[nodiscard]] auto data() const { return blocks_.data(); }

  [[nodiscard]] auto data() { return blocks_.data(); }

  [[nodiscard]] auto span() const {
    return BitVectorSpan<SpaceHost, TThreadSafety, decltype(data())>{data(), Base::size()};
  }

  [[nodiscard]] auto span() { return BitVectorSpan<SpaceHost, TThreadSafety, decltype(data())>{data(), Base::size()}; }

  [[nodiscard]] auto rawSpan() const { return span(); }

  [[nodiscard]] auto rawSpan() { return span(); }

private:
  using Base = BitVectorStorageAPI<BitVector<SpaceHost, TThreadSafety>, TThreadSafety>;
  friend Base;

  std::vector<typename Base::TBlock> blocks_;

  [[nodiscard]] const auto &storage() const { return blocks_; }

  [[nodiscard]] auto &storage() { return blocks_; }
};

template <typename TThreadSafety>
class BitVector<SpaceDevice, TThreadSafety>
    : public BitVectorStorageAPI<BitVector<SpaceDevice, TThreadSafety>, TThreadSafety> {
public:
  explicit BitVector(size_t n = 0) { Base::resize(n); }

  ARIA_COPY_MOVE_ABILITY(BitVector, default, default);

public:
  [[nodiscard]] auto data() const { return blocks_.data(); }

  [[nodiscard]] auto data() { return blocks_.data(); }

  [[nodiscard]] auto span() const {
    return BitVectorSpan<SpaceDevice, TThreadSafety, decltype(data())>{data(), Base::size()};
  }

  [[nodiscard]] auto span() {
    return BitVectorSpan<SpaceDevice, TThreadSafety, decltype(data())>{data(), Base::size()};
  }

  [[nodiscard]] auto rawSpan() const {
    return BitVectorSpan<SpaceDevice, TThreadSafety, decltype(thrust::raw_pointer_cast(data()))>{data(), Base::size()};
  }

  [[nodiscard]] auto rawSpan() {
    return BitVectorSpan<SpaceDevice, TThreadSafety, decltype(thrust::raw_pointer_cast(data()))>{data(), Base::size()};
  }

private:
  using Base = BitVectorStorageAPI<BitVector<SpaceDevice, TThreadSafety>, TThreadSafety>;
  friend Base;

  thrust::device_vector<typename Base::TBlock> blocks_;

  [[nodiscard]] const auto &storage() const { return blocks_; }

  [[nodiscard]] auto &storage() { return blocks_; }
};

} // namespace ARIA
