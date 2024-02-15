#include "ARIA/Scene/Behavior.h"
#include "ARIA/Scene/Object.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

class TestBehavior : public Behavior {
private:
  friend Object;

  using Base = Behavior;
  using Base::Base;
};

} // namespace

TEST(Behavior, Base) {
  Object &o = Object::Create();
  Behavior &b = o.AddComponent<TestBehavior>();
  EXPECT_EQ(&b, o.GetComponent<TestBehavior>());

  EXPECT_TRUE(b.enabled());
  b.enabled() = false;
  EXPECT_FALSE(b.enabled());
  b.enabled() = true;
  EXPECT_TRUE(b.enabled());
}

} // namespace ARIA
