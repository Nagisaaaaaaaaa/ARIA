#pragma once

#include "ARIA/ARIA.h"

namespace ARIA {

class Lifecycle {
public:
  virtual void Boot() = 0;

public:
  Lifecycle() = default;
  Lifecycle(const Lifecycle &) = delete;
  Lifecycle(Lifecycle &&) noexcept = delete;
  Lifecycle &operator=(const Lifecycle &) = delete;
  Lifecycle &operator=(Lifecycle &&) noexcept = delete;
  virtual ~Lifecycle() = default;
};

} // namespace ARIA
