#pragma once

#include "ARIA/Concurrency/Registry.h"

#include <cppcoro/task.hpp>

namespace ARIA {

class OMonoBehavior : public Registry<OMonoBehavior> {
public:
  [[nodiscard]] virtual cppcoro::task<> Evolve() = 0;

  [[nodiscard]] static Range<Iterator<OMonoBehavior>, Iterator<OMonoBehavior>> range() noexcept;

public:
  using Base = Registry<OMonoBehavior>;
  OMonoBehavior();

  OMonoBehavior(const OMonoBehavior &) = delete;
  OMonoBehavior(OMonoBehavior &&) noexcept = delete;
  OMonoBehavior &operator=(const OMonoBehavior &) = delete;
  OMonoBehavior &operator=(OMonoBehavior &&) noexcept = delete;

  ~OMonoBehavior() noexcept override;
};

} // namespace ARIA
