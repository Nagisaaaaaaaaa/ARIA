#pragma once

/// \file
/// \brief A `BitVector` is a policy-based owning vector containing bits, which
/// is similar to `std::vector<bool>`, but can be
/// host-storage or device-storage and thread-unsafe or thread-safe.
///
/// It is especially helpful when you want to save GPU memory, because
/// there are very few open-sourced GPU bit vector implementations.
//
//
//
//
//
#include "ARIA/detail/BitVectorImpl.h"

namespace ARIA {

/// \brief A `BitVector` is a policy-based owning vector containing bits, which
/// is similar to `std::vector<bool>`, but can be
/// host-storage or device-storage and thread-unsafe or thread-safe.
///
/// \example ```cpp
/// using Bits = BitVector<SpaceDevice, ThreadSafe>;
///
/// Bits bits(10);
/// size_t size = bits.size();
/// bits.resize(100);
///
/// bits.Fill(0);
/// bits.Clear(0);
/// bits.Flip(99);
///
/// bool bit0 = bits[99];
/// bits[99] = false;
/// bool bit1 = bits.at(99);
/// bits.at(99) = true;
///
/// SomeKernel<<<...>>>(bits.span());
/// ```
///
/// \warning `at(i)` and `operator[]` are never atomic even though the `ThreadSafe` policy is used.
/// Since setting a bit requires twice the efforts than filling, clearing, or flipping a bit,
/// developers should try to use `Fill`, `Clear`, and `Flip` instead.
using bit_vector::detail::BitVector;

//
//
//
/// \brief A `BitVectorSpan` is a policy-based non-owning view of a vector containing bits, which
/// is similar to `std::span<bool>`, but can be thread-unsafe or thread-safe.
///
/// \example ```cpp
/// using Bits = BitVector<SpaceDevice, ThreadSafe>;
///
/// Bits bits(100);
/// BitVectorSpan s = bits.span();
///
/// s.Fill(0);
/// s.Clear(0);
/// s.Flip(99);
///
/// bool bit0 = s[99];
/// s[99] = false;
/// bool bit1 = s.at(99);
/// s.at(99) = true;
///
/// SomeKernel<<<...>>>(s);
/// ```
///
/// \warning `at(i)` and `operator[]` are never atomic even though the `ThreadSafe` policy is used.
/// Since setting a bit requires twice the efforts than filling, clearing, or flipping a bit,
/// developers should try to use `Fill`, `Clear`, and `Flip` instead.
using bit_vector::detail::BitVectorSpan;

} // namespace ARIA
