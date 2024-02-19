#include "ARIA/DisjointSet.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(DisjointSet, Base) {
  DisjointSet<ThreadSafe, std::vector<int>> disjointSet;
}

} // namespace ARIA
