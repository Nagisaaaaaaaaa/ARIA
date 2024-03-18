#pragma once

#include "ARIA/Property.h"

#include <cuda/atomic>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace ARIA {

namespace bit_vector::detail {

//! For future developers: Please read the following comments to help you understand the codes.
//! ! `BitVector` should be implemented similar to `std::vector<bool>`, but
//! should also support device vectors and thread-safety.
//! So, there are 4 variants of `BitVector`s:
//!   1. Host + thread-unsafe,
//!   2. Host + thread-safe,
//!   3. Device + thread-unsafe,
//!   4. Device + thread-safe.

template <typename TDerived>
class BitVectorCRTPBase {
protected:
  using TBlock = uint;
  static constexpr uint nBitsPerBlock = sizeof(TBlock) * 8;

  [[nodiscard]] ARIA_HOST_DEVICE const TDerived &derived() const { return *static_cast<const TDerived *>(this); }

  [[nodiscard]] ARIA_HOST_DEVICE TDerived &derived() { return *static_cast<TDerived *>(this); }
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

  ARIA_HOST_DEVICE friend void swap(BitVectorSpanAPI &lhs, BitVectorSpanAPI &rhs) ARIA_NOEXCEPT {
    using std::swap;
    swap(lhs.nBits_, rhs.nBits_);
  }

  ARIA_HOST_DEVICE BitVectorSpanAPI(BitVectorSpanAPI &&other) ARIA_NOEXCEPT : BitVectorSpanAPI() { swap(*this, other); }

  ARIA_HOST_DEVICE BitVectorSpanAPI &operator=(BitVectorSpanAPI &&other) ARIA_NOEXCEPT {
    swap(*this, other);
    return *this;
  }

public:
  ARIA_PROP(public, public, ARIA_HOST_DEVICE, bool, at, size_t);

public:
  [[nodiscard]] ARIA_HOST_DEVICE auto operator[](size_t i) const { return at(i); }

  [[nodiscard]] ARIA_HOST_DEVICE auto operator[](size_t i) { return at(i); }

  ARIA_HOST_DEVICE TDerived &Fill(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    FillBit(derived().data()[iBlocks], iBits);
    return derived();
  }

  ARIA_HOST_DEVICE TDerived &Clear(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    ClearBit(derived().data()[iBlocks], iBits);
    return derived();
  }

  ARIA_HOST_DEVICE TDerived &Flip(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    FlipBit(derived().data()[iBlocks], iBits);
    return derived();
  }

  [[nodiscard]] ARIA_HOST_DEVICE size_t size() const { return nBits_; }

private:
  // Iterators fwd.
  template <typename TDerivedMaybeConst>
  class Iterator;

public:
  [[nodiscard]] ARIA_HOST_DEVICE auto begin() const { return Iterator<const TDerived>{derived(), 0}; }

  [[nodiscard]] ARIA_HOST_DEVICE auto begin() { return Iterator<TDerived>{derived(), 0}; }

  [[nodiscard]] ARIA_HOST_DEVICE auto end() const { return Iterator<const TDerived>{derived(), nBits_}; }

  [[nodiscard]] ARIA_HOST_DEVICE auto end() { return Iterator<TDerived>{derived(), nBits_}; }

  [[nodiscard]] ARIA_HOST_DEVICE auto cbegin() const { return begin(); }

  [[nodiscard]] ARIA_HOST_DEVICE auto cend() const { return end(); }

protected:
  size_t nBits_ = 0;

  [[nodiscard]] ARIA_HOST_DEVICE std::pair<size_t, uint> i2iBlocksAndiBits(size_t i) const {
    ARIA_ASSERT(i < nBits_, "The given bit index should be smaller than the total number of bits");

    size_t iBlocks = i / nBitsPerBlock;
    uint iBits = i % nBitsPerBlock;
    return {iBlocks, iBits};
  }

  [[nodiscard]] ARIA_HOST_DEVICE static bool GetBit(const TBlock &block, uint iBits) {
    return static_cast<bool>((block >> iBits) & TBlock{1});
  }

  [[nodiscard]] ARIA_HOST_DEVICE static bool GetBit(const thrust::device_reference<const TBlock> &block, uint iBits) {
    return static_cast<bool>((block >> iBits) & TBlock{1});
  }

  [[nodiscard]] ARIA_HOST_DEVICE static bool GetBit(const thrust::device_reference<TBlock> &block, uint iBits) {
    return static_cast<bool>((block >> iBits) & TBlock{1});
  }

  ARIA_HOST_DEVICE static void FillBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadSafety, ThreadUnsafe>) {
      block |= (TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadSafety, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_or(TBlock{1} << iBits);
    }
  }

  ARIA_HOST_DEVICE static void FillBit(const thrust::device_reference<TBlock> &block, uint iBits) {
    FillBit(thrust::raw_reference_cast(block), iBits);
  }

  ARIA_HOST_DEVICE static void ClearBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadSafety, ThreadUnsafe>) {
      block &= ~(TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadSafety, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_and(~(TBlock{1} << iBits));
    }
  }

  ARIA_HOST_DEVICE static void ClearBit(const thrust::device_reference<TBlock> &block, uint iBits) {
    ClearBit(thrust::raw_reference_cast(block), iBits);
  }

  ARIA_HOST_DEVICE static void FlipBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadSafety, ThreadUnsafe>) {
      block ^= (TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadSafety, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_xor(TBlock{1} << iBits);
    }
  }

  ARIA_HOST_DEVICE static void FlipBit(const thrust::device_reference<TBlock> &block, uint iBits) {
    FlipBit(thrust::raw_reference_cast(block), iBits);
  }

  ARIA_HOST_DEVICE static void SetBit(TBlock &block, uint iBits, bool bit) {
    TBlock mask = TBlock{1} << iBits;
    block = (block & ~mask) | (TBlock{bit} << iBits);
  }

  ARIA_HOST_DEVICE static void SetBit(thrust::device_reference<TBlock> block, uint iBits, bool bit) {
    TBlock mask = TBlock{1} << iBits;
    block = (block & ~mask) | (TBlock{bit} << iBits);
  }

  [[nodiscard]] ARIA_HOST_DEVICE bool ARIA_PROP_IMPL(at)(size_t i) const {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    return GetBit(derived().data()[iBlocks], iBits);
  }

  ARIA_HOST_DEVICE void ARIA_PROP_IMPL(at)(size_t i, bool value) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    SetBit(derived().data()[iBlocks], iBits, value);
  }

private:
  // Iterators implementation.
  template <typename TDerivedMaybeConst>
  class Iterator {
  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = bool;
    // using reference = decltype(std::declval<BitVectorSpanAPI>()[0]);
    // using pointer = ArrowProxy<reference>;

  public:
    //! The ranges library requires that iterators should be default constructable.
    Iterator() noexcept = default;

    ARIA_HOST_DEVICE explicit Iterator(TDerivedMaybeConst &bitVector, size_t i) noexcept
        : bitVector_(&bitVector), i_(i) {}

    ARIA_COPY_MOVE_ABILITY(Iterator, default, default);

  public:
    ARIA_HOST_DEVICE auto operator*() const { return (*bitVector_)[i_]; }

    ARIA_HOST_DEVICE auto operator*() { return (*bitVector_)[i_]; }

    ARIA_HOST_DEVICE auto operator->() const noexcept { return ArrowProxy((*bitVector_)[i_]); }

    ARIA_HOST_DEVICE auto operator->() noexcept { return ArrowProxy((*bitVector_)[i_]); }

    // ++it
    ARIA_HOST_DEVICE Iterator &operator++() {
      ++i_;
      return *this;
    }

    // it++
    ARIA_HOST_DEVICE Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    // --it
    ARIA_HOST_DEVICE Iterator &operator--() {
      --i_;
      return *this;
    }

    // it--
    ARIA_HOST_DEVICE Iterator operator--(int) {
      Iterator tmp = *this;
      --(*this);
      return tmp;
    }

    ARIA_HOST_DEVICE friend bool operator==(const Iterator &a, const Iterator &b) noexcept {
      return a.bitVector_ == b.bitVector_ && a.i_ == b.i_;
    }

    ARIA_HOST_DEVICE friend bool operator!=(const Iterator &a, const Iterator &b) noexcept { return !operator==(a, b); }

  private:
    TDerivedMaybeConst *bitVector_ = nullptr;
    size_t i_ = 0;
  };
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
  [[nodiscard]] ARIA_HOST_DEVICE auto data() const { return p_; }

  [[nodiscard]] ARIA_HOST_DEVICE auto data() { return p_; }

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

  thrust::host_vector<typename Base::TBlock> blocks_;

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
    return BitVectorSpan<SpaceDevice, TThreadSafety, decltype(thrust::raw_pointer_cast(data()))>{
        thrust::raw_pointer_cast(data()), Base::size()};
  }

  [[nodiscard]] auto rawSpan() {
    return BitVectorSpan<SpaceDevice, TThreadSafety, decltype(thrust::raw_pointer_cast(data()))>{
        thrust::raw_pointer_cast(data()), Base::size()};
  }

private:
  using Base = BitVectorStorageAPI<BitVector<SpaceDevice, TThreadSafety>, TThreadSafety>;
  friend Base;

  thrust::device_vector<typename Base::TBlock> blocks_;

  [[nodiscard]] const auto &storage() const { return blocks_; }

  [[nodiscard]] auto &storage() { return blocks_; }
};

} // namespace bit_vector::detail

} // namespace ARIA
