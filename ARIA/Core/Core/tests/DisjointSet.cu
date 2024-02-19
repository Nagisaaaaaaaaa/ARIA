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
  //   1 2 2 2 2 x
  //   1 1 2 2 2
  //   0 1 2 2 2
  //   0 1 1 2 2
  //   0 0 1 1 1
  //   y
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

  // Supporting functions.
  auto crd2Idx = [&](auto x, auto y) { return flags.layout()(x, y); };
  auto printLabels = [&]() {
    for (int y = 0; y < disjointSet.labels().size<1>(); ++y) {
      for (int x = 0; x < disjointSet.labels().size<0>(); ++x) {
        fmt::print("{} ", disjointSet.labels()(x, y));
      }
      fmt::print("\n");
    }
  };
  auto expectLabel = [&](auto x, auto y, auto label) { EXPECT_EQ(disjointSet.labels()(x, y), label); };
  // clang-format off
  auto expectLabels = [&](auto v00, auto v10, auto v20, auto v30, auto v40,
                          auto v01, auto v11, auto v21, auto v31, auto v41,
                          auto v02, auto v12, auto v22, auto v32, auto v42,
                          auto v03, auto v13, auto v23, auto v33, auto v43,
                          auto v04, auto v14, auto v24, auto v34, auto v44) {
    expectLabel(0, 0, v00); expectLabel(1, 0, v10); expectLabel(2, 0, v20); expectLabel(3, 0, v30); expectLabel(4, 0, v40);
    expectLabel(0, 1, v01); expectLabel(1, 1, v11); expectLabel(2, 1, v21); expectLabel(3, 1, v31); expectLabel(4, 1, v41);
    expectLabel(0, 2, v02); expectLabel(1, 2, v12); expectLabel(2, 2, v22); expectLabel(3, 2, v32); expectLabel(4, 2, v42);
    expectLabel(0, 3, v03); expectLabel(1, 3, v13); expectLabel(2, 3, v23); expectLabel(3, 3, v33); expectLabel(4, 3, v43);
    expectLabel(0, 4, v04); expectLabel(1, 4, v14); expectLabel(2, 4, v24); expectLabel(3, 4, v34); expectLabel(4, 4, v44);
  };
  // clang-format on

  // Union.
  disjointSet.Union(crd2Idx(0, 4), crd2Idx(1, 4));
  expectLabels(0, 1, 2, 3, 4,      //
               5, 6, 7, 8, 9,      //
               10, 11, 12, 13, 14, //
               15, 16, 17, 18, 19, //
               20, 20, 22, 23, 24);
}

} // namespace ARIA
