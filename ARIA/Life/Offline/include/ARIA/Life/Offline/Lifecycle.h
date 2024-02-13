#pragma once

#include "ARIA/Life/Lifecycle.h"

namespace ARIA {

class OLifecycle final : public Lifecycle {
public:
  void Boot() final;

public:
  OLifecycle() = default;
  OLifecycle(const OLifecycle &) = delete;
  OLifecycle(OLifecycle &&) noexcept = delete;
  OLifecycle &operator=(const OLifecycle &) = delete;
  OLifecycle &operator=(OLifecycle &&) noexcept = delete;
  ~OLifecycle() final = default;
};

} // namespace ARIA
