#include "ARIA/DisjointSet.h"
#include "ARIA/TensorVector.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

enum Flag : int { G = 0, I = 1, L = 2 };

} // namespace

TEST(DisjointSet, Base) {
  using Volume = TensorVector<int, C<2>, SpaceHost>;

  // Test case:
  //   1 2 2 2 2
  //   1 1 2 2 2
  //   0 1 2 2 2
  //   0 1 1 2 2
  //   0 0 1 1 1
  Volume flags{make_layout_major(5, 5)};
  // clang-format off
  flags(0, 0) = 1; flags(1, 0) = 2; flags(2, 0) = 2; flags(3, 0) = 2; flags(4, 0) = 2;
  flags(0, 1) = 1; flags(1, 1) = 1; flags(2, 1) = 2; flags(3, 1) = 2; flags(4, 1) = 2;
  flags(0, 2) = 0; flags(1, 2) = 1; flags(2, 2) = 2; flags(3, 2) = 2; flags(4, 2) = 2;
  flags(0, 3) = 0; flags(1, 3) = 1; flags(2, 3) = 1; flags(3, 3) = 2; flags(4, 3) = 2;
  flags(0, 4) = 0; flags(1, 4) = 0; flags(2, 4) = 1; flags(3, 4) = 1; flags(4, 4) = 1;
  // clang-format on

  // Initialize.
  DisjointSet<ThreadSafe, Volume> disjointSet(Volume{make_layout_major(5, 5)});
  for (int y = 0; y < disjointSet.labels().size<1>(); ++y)
    for (int x = 0; x < disjointSet.labels().size<0>(); ++x)
      disjointSet.labels()(x, y) = disjointSet.labels().layout()(x, y);

  // Union.
  // TODO: Implement this.
}

} // namespace ARIA
