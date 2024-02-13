#pragma once

#include "ARIA/Life/Lifecycle.h"

namespace ARIA {

class RLifecycle final : public Lifecycle {
public:
  void Boot() final;

public:
  RLifecycle() = default;
  RLifecycle(const RLifecycle &) = delete;
  RLifecycle(RLifecycle &&) noexcept = delete;
  RLifecycle &operator=(const RLifecycle &) = delete;
  RLifecycle &operator=(RLifecycle &&) noexcept = delete;
  ~RLifecycle() final = default;
};

} // namespace ARIA
