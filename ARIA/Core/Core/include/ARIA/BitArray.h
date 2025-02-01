#pragma once

/// \file
/// \brief A `BitArray` is a policy-based owning array containing bits, which
/// is similar to `std::bitset`, but can be
/// used at host or device sides and thread-unsafe or thread-safe.
///
/// It is especially helpful when you want to save GPU memory, because
/// there are very few open-sourced GPU bit array implementations.
//
//
//
//
//
#include "ARIA/Property.h"

#include <cuda/atomic>
#include <cuda/std/array>

namespace ARIA {

/// \brief A `BitArray` is a policy-based owning array containing bits, which
/// is similar to `std::bitset`, but can be
/// used at host or device sides and thread-unsafe or thread-safe.
///
/// \example ```cpp
/// using Bits = BitArray<100, ThreadSafe>;
///
/// Bits bits;
/// constexpr size_t size = bits.size();
///
/// bits.Fill(0);
/// bits.Clear(0);
/// bits.Flip(99);
///
/// bool bit0 = bits[99];
/// bits[99] = false;
/// bool bit1 = bits.at(99);
/// bits.at(99) = true;
/// ```
///
/// \warning `at(i)` and `operator[]` are never atomic even though the `ThreadSafe` policy is used.
/// Since setting a bit requires twice the efforts than filling, clearing, or flipping a bit,
/// developers should try to use `Fill`, `Clear`, and `Flip` instead.
///
/// \note Implementation of `BitArray` is much simpler than `BitVector`, so,
/// codes of `BitArray` are not explained in detail.
/// If you are interested in the implementation details, please
/// first read the implementation of `BitVector`, see `BitVector.h`.
template <size_t nBits, typename TThreadSafety>
class BitArray {
public:
  /// \brief Construct a `BitArray` with all bits cleared.
  ARIA_HOST_DEVICE BitArray() { storage_.fill(0); }

  //! Copy is allowed but move is forbidden because it is unable to clear the array being moved.
  //! Allowing move will make the semantics weird.
  //! So, users should always use copy instead of move.
  ARIA_COPY_MOVE_ABILITY(BitArray, default, delete);

  //
  //
  //
public:
  /// \brief Access the i^th bit.
  ///
  /// \warning This method is never atomic even though the `ThreadSafe` policy is used.
  ARIA_PROP(public, public, ARIA_HOST_DEVICE, bool, at, size_t);

public:
  /// \brief Access the i^th bit.
  ///
  /// \warning This method is never atomic even though the `ThreadSafe` policy is used.
  [[nodiscard]] ARIA_HOST_DEVICE auto operator[](size_t i) const { return at(i); }

  /// \brief Access the i^th bit.
  ///
  /// \warning This method is never atomic even though the `ThreadSafe` policy is used.
  [[nodiscard]] ARIA_HOST_DEVICE auto operator[](size_t i) { return at(i); }

  //
  //
  //
  /// \brief Fill the i^th bit.
  ///
  /// \warning This method is atomic when the `ThreadSafe` policy is used, while
  /// non-atomic when the `ThreadUnsafe` policy is used.
  ARIA_HOST_DEVICE BitArray &Fill(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    FillBit(storage_[iBlocks], iBits);
    return *this;
  }

  /// \brief Clear the i^th bit.
  ///
  /// \warning This method is atomic when the `ThreadSafe` policy is used, while
  /// non-atomic when the `ThreadUnsafe` policy is used.
  ARIA_HOST_DEVICE BitArray &Clear(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    ClearBit(storage_[iBlocks], iBits);
    return *this;
  }

  /// \brief Flip the i^th bit.
  ///
  /// \warning This method is atomic when the `ThreadSafe` policy is used, while
  /// non-atomic when the `ThreadUnsafe` policy is used.
  ARIA_HOST_DEVICE BitArray &Flip(size_t i) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    FlipBit(storage_[iBlocks], iBits);
    return *this;
  }

  /// \brief Get the number of bits.
  [[nodiscard]] ARIA_HOST_DEVICE consteval size_t size() const { return nBits; }

  //
  //
  //
private:
  template <typename BitArrayMaybeConst>
  class Iterator;

public:
  [[nodiscard]] ARIA_HOST_DEVICE auto begin() const { return Iterator<const BitArray>{*this, 0}; }

  [[nodiscard]] ARIA_HOST_DEVICE auto begin() { return Iterator<BitArray>{*this, 0}; }

  [[nodiscard]] ARIA_HOST_DEVICE auto end() const { return Iterator<const BitArray>{*this, nBits}; }

  [[nodiscard]] ARIA_HOST_DEVICE auto end() { return Iterator<BitArray>{*this, nBits}; }

  [[nodiscard]] ARIA_HOST_DEVICE auto cbegin() const { return begin(); }

  [[nodiscard]] ARIA_HOST_DEVICE auto cend() const { return end(); }

  //
  //
  //
  //
  //
  //
  //
  //
  //
private:
  // A "block" is a minimum unit of storage, where atomic operations can be performed on.
  // The binary representations of blocks are the bits we are interested in.
  using TBlock = uint;

  // Number of bits per block.
  static constexpr uint nBitsPerBlock = sizeof(TBlock) * 8;

  // Number of blocks.
  static constexpr size_t nBlocks = (nBits + nBitsPerBlock - 1) / nBitsPerBlock;

  cuda::std::array<TBlock, nBlocks> storage_;

private:
  // Definitions: `i`: bit idx, `iBlocks`: block idx, `iBits`: bit idx in block.
  [[nodiscard]] ARIA_HOST_DEVICE std::pair<size_t, uint> i2iBlocksAndiBits(size_t i) const {
    ARIA_ASSERT(i < nBits, "The given bit index should be smaller than the total number of bits");

    size_t iBlocks = i / nBitsPerBlock;
    uint iBits = i % nBitsPerBlock;
    return {iBlocks, iBits};
  }

  // Get the `iBits`^th bit in `block`.
  [[nodiscard]] ARIA_HOST_DEVICE static bool GetBit(const TBlock &block, uint iBits) {
    return static_cast<bool>((block >> iBits) & TBlock{1});
  }

  // Fill the `iBits`^th bit in `block`.
  ARIA_HOST_DEVICE static void FillBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadSafety, ThreadUnsafe>) {
      block |= (TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadSafety, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_or(TBlock{1} << iBits);
    }
  }

  // Clear the `iBits`^th bit in `block`.
  ARIA_HOST_DEVICE static void ClearBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadSafety, ThreadUnsafe>) {
      block &= ~(TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadSafety, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_and(~(TBlock{1} << iBits));
    }
  }

  // Flip the `iBits`^th bit in `block`.
  ARIA_HOST_DEVICE static void FlipBit(TBlock &block, uint iBits) {
    if constexpr (std::is_same_v<TThreadSafety, ThreadUnsafe>) {
      block ^= (TBlock{1} << iBits);
    } else if constexpr (std::is_same_v<TThreadSafety, ThreadSafe>) {
      cuda::atomic_ref atomicBlock{block};
      atomicBlock.fetch_xor(TBlock{1} << iBits);
    }
  }

  // Set the `iBits`^th bit in `block`.
  //! Now, you may have understand why
  //! `at(i)` and `operator[]` cannot be trivially implemented as atomic.
  ARIA_HOST_DEVICE static void SetBit(TBlock &block, uint iBits, bool bit) {
    TBlock mask = TBlock{1} << iBits;
    block = (block & ~mask) | (TBlock{bit} << iBits);
  }

  [[nodiscard]] ARIA_HOST_DEVICE bool ARIA_PROP_GETTER(at)(size_t i) const {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    return GetBit(storage_[iBlocks], iBits);
  }

  ARIA_HOST_DEVICE void ARIA_PROP_SETTER(at)(size_t i, bool value) {
    auto [iBlocks, iBits] = i2iBlocksAndiBits(i);
    SetBit(storage_[iBlocks], iBits, value);
  }

private:
  template <typename BitArrayMaybeConst>
  class Iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = bool;

  public:
    Iterator() noexcept = default;

    ARIA_HOST_DEVICE explicit Iterator(BitArrayMaybeConst &v, size_t i) noexcept : v_(&v), i_(i) {}

    ARIA_COPY_MOVE_ABILITY(Iterator, default, default);

  public:
    ARIA_HOST_DEVICE auto operator*() const { return (*v_)[i_]; }

    ARIA_HOST_DEVICE auto operator*() { return (*v_)[i_]; }

    ARIA_HOST_DEVICE auto operator->() const noexcept { return ArrowProxy((*v_)[i_]); }

    ARIA_HOST_DEVICE auto operator->() noexcept { return ArrowProxy((*v_)[i_]); }

    ARIA_HOST_DEVICE friend bool operator==(const Iterator &a, const Iterator &b) noexcept {
      return a.v_ == b.v_ && a.i_ == b.i_;
    }

    ARIA_HOST_DEVICE friend bool operator!=(const Iterator &a, const Iterator &b) noexcept { return !operator==(a, b); }

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

    ARIA_HOST_DEVICE Iterator operator+(size_t n) const {
      Iterator temp = *this;
      return (temp += n);
    };

    ARIA_HOST_DEVICE Iterator operator-(size_t n) const {
      Iterator temp = *this;
      return (temp -= n);
    };

    ARIA_HOST_DEVICE friend Iterator operator+(size_t n, const Iterator &it) { return it + n; }

    ARIA_HOST_DEVICE friend Iterator operator-(size_t n, const Iterator &it) { return it - n; }

    ARIA_HOST_DEVICE Iterator &operator+=(size_t n) {
      i_ += n;
      return *this;
    }

    ARIA_HOST_DEVICE Iterator &operator-=(size_t n) {
      i_ -= n;
      return *this;
    }

    ARIA_HOST_DEVICE auto operator[](size_t n) const { return (*v_)[i_ + n]; }

    ARIA_HOST_DEVICE friend difference_type operator-(const Iterator &lhs, const Iterator &rhs) noexcept {
      return lhs.i_ - rhs.i_;
    }

    ARIA_HOST_DEVICE friend bool operator<(const Iterator &lhs, const Iterator &rhs) noexcept {
      return lhs.i_ < rhs.i_;
    }

    ARIA_HOST_DEVICE friend bool operator<=(const Iterator &lhs, const Iterator &rhs) noexcept {
      return lhs.i_ <= rhs.i_;
    }

    ARIA_HOST_DEVICE friend bool operator>(const Iterator &lhs, const Iterator &rhs) noexcept {
      return lhs.i_ > rhs.i_;
    }

    ARIA_HOST_DEVICE friend bool operator>=(const Iterator &lhs, const Iterator &rhs) noexcept {
      return lhs.i_ >= rhs.i_;
    }

  private:
    BitArrayMaybeConst *v_ = nullptr;
    size_t i_ = 0;
  };
};

} // namespace ARIA
