#include "ARIA/Scene/Components/Transform.h"
#include "ARIA/Scene/Object.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {} // namespace

TEST(Component, Base) {
  // Transform.
  {
    Object &o = Object::Create();
    Transform &t = o.transform();
    {
      Object &to = t.object();
      EXPECT_EQ(o, to);
    }
    {
      Object &toto = t.object().transform().object();
      EXPECT_EQ(o, toto);
    }
    {
      Transform &ott = o.transform().transform();
      EXPECT_EQ(t, ott);
    }
    {
      Transform &otot = o.transform().object().transform();
      EXPECT_EQ(t, otot);
    }
  }
}

} // namespace ARIA
