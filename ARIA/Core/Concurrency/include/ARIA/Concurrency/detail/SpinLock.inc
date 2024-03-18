#pragma once

namespace ARIA {

ARIA_HOST_DEVICE inline void SpinLock::lock() noexcept {
  for (;;) {
    // Optimistically assume the lock is free on the first try.
    if (!lock_.exchange(true, cuda::std::memory_order_acquire)) {
      return;
    }
    // Wait for lock to be released without generating cache misses.
    while (lock_.load(cuda::std::memory_order_relaxed)) {
      // Issue X86 PAUSE or ARM YIELD instruction to
      // reduce contention between hyper-threads.
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
}

ARIA_HOST_DEVICE inline void SpinLock::unlock() noexcept {
  lock_.store(false, cuda::std::memory_order_release);
}

[[nodiscard]] ARIA_HOST_DEVICE inline bool SpinLock::try_lock() noexcept {
  // First do a relaxed load to check if lock is free in order to prevent
  // unnecessary cache misses if someone does `while(!try_lock())`.
  return !lock_.load(cuda::std::memory_order_relaxed) && !lock_.exchange(true, cuda::std::memory_order_acquire);
}

} // namespace ARIA
