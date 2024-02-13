#pragma once

#include "ARIA/Scene/Behavior.h"

namespace ARIA {

class RMonoBehavior : public Behavior {
public:
  virtual void Start(){};
  virtual void Update(){};
  virtual void FixedUpdate(){};

protected:
  friend Object;

  using Base = Behavior;
  using Base::Base;

public:
  RMonoBehavior(const RMonoBehavior &) = delete;
  RMonoBehavior(RMonoBehavior &&) noexcept = delete;
  RMonoBehavior &operator=(const RMonoBehavior &) = delete;
  RMonoBehavior &operator=(RMonoBehavior &&) noexcept = delete;
  ~RMonoBehavior() override = default;
};

} // namespace ARIA
