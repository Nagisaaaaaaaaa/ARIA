#pragma once

#include "ARIA/ARIA.h"

namespace ARIA {

/// \brief A broken mutex always fail to lock and unlock.
/// It just does nothing.
class BrokenMutex {
public:
  BrokenMutex() = default;

  ARIA_COPY_MOVE_ABILITY(BrokenMutex, delete, delete);

public:
  void lock() noexcept {
    // Do nothing.
  }

  void unlock() noexcept {
    // Do nothing.
  }

  bool try_lock() noexcept {
    // Do nothing.
  }
};

} // namespace ARIA
