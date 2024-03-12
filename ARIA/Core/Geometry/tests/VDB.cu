#include "ARIA/VDB.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(VDB, Base) {
  VDB<float, 2, SpaceDevice> vdb;
}

} // namespace ARIA
