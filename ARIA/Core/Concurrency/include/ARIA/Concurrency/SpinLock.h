#pragma once

/// \file
/// \brief This file provides a fully optimized spin lock implementation.
///
/// The spin lock structure is a low-level, mutual-exclusion synchronization primitive that
/// spins while it waits to acquire a lock.
/// On multi-core computers, when wait times are expected to be short and when contention ie minimal,
/// spin lock can perform better than other kinds of locks.

//
//
//
//
//
#include "ARIA/ARIA.h"

#include <cuda/std/atomic>

namespace ARIA {

/// \brief The spin lock structure is a low-level, mutual-exclusion synchronization primitive that
/// spins while it waits to acquire a lock.
/// On multi-core computers, when wait times are expected to be short and when contention ie minimal,
/// spin lock can perform better than other kinds of locks.
///
/// \example ```cpp
/// static constinit SpinLock lock{};
/// static constinit size_t count = 0;
///
/// void Inc(size_t n) {
///   for (size_t i = 0; i < n; ++i) {
///     std::lock_guard guard(lock);
///     ++count;
///   }
/// }
///
/// auto t = std::jthread{Inc, 1000};
/// ```
///
/// \details This spin lock implementation is based on https://rigtorp.se/spinlock/.
class SpinLock {
public:
  /// \brief Locks the mutex, blocks if the mutex is not available.
  ARIA_HOST_DEVICE inline void lock() noexcept;

  /// \brief Unlocks the mutex.
  ARIA_HOST_DEVICE inline void unlock() noexcept;

  /// \brief Tries to lock the mutex, returns if the mutex is not available.
  [[nodiscard]] ARIA_HOST_DEVICE inline bool try_lock() noexcept;

  //
public:
  SpinLock() = default;

  ARIA_COPY_MOVE_ABILITY(SpinLock, delete, delete);

  //
  //
  //
private:
  cuda::std::atomic<bool> lock_{false};
};

} // namespace ARIA

//
//
//
//
//
#include "ARIA/Concurrency/detail/SpinLock.inc"
