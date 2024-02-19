#include "ARIA/DisjointSet.h"
#include "ARIA/Launcher.h"
#include "ARIA/TensorVector.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

template <typename TThreadUnsafeOrSafe, typename SpaceHostOrDevice>
void TestCPU() {
  using Volume = TensorVector<int, C<2>, SpaceHostOrDevice>;

  enum Flag : int { G = 0, I = 1, L = 2 };

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
  DisjointSet<TThreadUnsafeOrSafe, Volume> disjointSet(Volume{make_layout_major(5, 5)});
  for (int y = 0; y < disjointSet.nodes().size<1>(); ++y)
    for (int x = 0; x < disjointSet.nodes().size<0>(); ++x)
      disjointSet.nodes()(x, y) = disjointSet.nodes().layout()(x, y);

  // Supporting functions.
  auto crd2Idx = [&](auto x, auto y) { return flags.layout()(x, y); };
#if 0
  auto printNodes = [&]() {
    for (int y = 0; y < disjointSet.nodes().size<1>(); ++y) {
      for (int x = 0; x < disjointSet.nodes().size<0>(); ++x) {
        fmt::print("{} ", disjointSet.nodes()(x, y));
      }
      fmt::print("\n");
    }
  };
#endif
  auto expectNode = [&](auto x, auto y, auto node) {
    EXPECT_EQ(disjointSet.FindAndCompress(crd2Idx(x, y)), node);
    EXPECT_EQ(disjointSet.Find(crd2Idx(x, y)), node);
    EXPECT_EQ(disjointSet.nodes()(x, y), node);
  };
  // clang-format off
  auto expectNodes = [&](auto v00, auto v10, auto v20, auto v30, auto v40,
                          auto v01, auto v11, auto v21, auto v31, auto v41,
                          auto v02, auto v12, auto v22, auto v32, auto v42,
                          auto v03, auto v13, auto v23, auto v33, auto v43,
                          auto v04, auto v14, auto v24, auto v34, auto v44) {
    expectNode(0, 0, v00); expectNode(1, 0, v10); expectNode(2, 0, v20); expectNode(3, 0, v30); expectNode(4, 0, v40);
    expectNode(0, 1, v01); expectNode(1, 1, v11); expectNode(2, 1, v21); expectNode(3, 1, v31); expectNode(4, 1, v41);
    expectNode(0, 2, v02); expectNode(1, 2, v12); expectNode(2, 2, v22); expectNode(3, 2, v32); expectNode(4, 2, v42);
    expectNode(0, 3, v03); expectNode(1, 3, v13); expectNode(2, 3, v23); expectNode(3, 3, v33); expectNode(4, 3, v43);
    expectNode(0, 4, v04); expectNode(1, 4, v14); expectNode(2, 4, v24); expectNode(3, 4, v34); expectNode(4, 4, v44);
  };
  // clang-format on

  // Simple union.
  expectNodes(0, 1, 2, 3, 4,      //
              5, 6, 7, 8, 9,      //
              10, 11, 12, 13, 14, //
              15, 16, 17, 18, 19, //
              20, 21, 22, 23, 24);

  disjointSet.Union(crd2Idx(0, 4), crd2Idx(0, 4));
  expectNodes(0, 1, 2, 3, 4,      //
              5, 6, 7, 8, 9,      //
              10, 11, 12, 13, 14, //
              15, 16, 17, 18, 19, //
              20, 21, 22, 23, 24);

  disjointSet.Union(crd2Idx(0, 4), crd2Idx(1, 4));
  expectNodes(0, 1, 2, 3, 4,      //
              5, 6, 7, 8, 9,      //
              10, 11, 12, 13, 14, //
              15, 16, 17, 18, 19, //
              20, 20, 22, 23, 24);

  disjointSet.Union(crd2Idx(0, 3), crd2Idx(0, 2));
  expectNodes(0, 1, 2, 3, 4,      //
              5, 6, 7, 8, 9,      //
              10, 11, 12, 13, 14, //
              10, 16, 17, 18, 19, //
              20, 20, 22, 23, 24);

  disjointSet.Union(crd2Idx(1, 4), crd2Idx(0, 2));
  expectNodes(0, 1, 2, 3, 4,      //
              5, 6, 7, 8, 9,      //
              10, 11, 12, 13, 14, //
              10, 16, 17, 18, 19, //
              10, 10, 22, 23, 24);

  disjointSet.Union(crd2Idx(1, 4), crd2Idx(1, 4));
  expectNodes(0, 1, 2, 3, 4,      //
              5, 6, 7, 8, 9,      //
              10, 11, 12, 13, 14, //
              10, 16, 17, 18, 19, //
              10, 10, 22, 23, 24);

  // Complex union.
  for (int y = 0; y < disjointSet.nodes().size<1>(); ++y)
    for (int x = 0; x < disjointSet.nodes().size<0>(); ++x)
      if (flags(x, y) == L) {
        if (x - 1 >= 0 && flags(x - 1, y) == L)
          disjointSet.Union(crd2Idx(x - 1, y), crd2Idx(x, y));
        if (x + 1 < disjointSet.nodes().size<0>() && flags(x + 1, y) == L)
          disjointSet.Union(crd2Idx(x + 1, y), crd2Idx(x, y));
        if (y - 1 >= 0 && flags(x, y - 1) == L)
          disjointSet.Union(crd2Idx(x, y - 1), crd2Idx(x, y));
        if (y + 1 < disjointSet.nodes().size<1>() && flags(x, y + 1) == L)
          disjointSet.Union(crd2Idx(x, y + 1), crd2Idx(x, y));
      }
  expectNodes(0, 1, 1, 1, 1,    //
              5, 6, 1, 1, 1,    //
              10, 11, 1, 1, 1,  //
              10, 16, 17, 1, 1, //
              10, 10, 22, 23, 24);

  for (int y = 0; y < disjointSet.nodes().size<1>(); ++y)
    for (int x = 0; x < disjointSet.nodes().size<0>(); ++x)
      if (flags(x, y) == I) {
        if (x - 1 >= 0 && flags(x - 1, y) == I)
          disjointSet.Union(crd2Idx(x - 1, y), crd2Idx(x, y));
        if (x + 1 < disjointSet.nodes().size<0>() && flags(x + 1, y) == I)
          disjointSet.Union(crd2Idx(x + 1, y), crd2Idx(x, y));
        if (y - 1 >= 0 && flags(x, y - 1) == I)
          disjointSet.Union(crd2Idx(x, y - 1), crd2Idx(x, y));
        if (y + 1 < disjointSet.nodes().size<1>() && flags(x, y + 1) == I)
          disjointSet.Union(crd2Idx(x, y + 1), crd2Idx(x, y));
      }
  expectNodes(0, 1, 1, 1, 1,  //
              0, 0, 1, 1, 1,  //
              10, 0, 1, 1, 1, //
              10, 0, 0, 1, 1, //
              10, 10, 0, 0, 0);

  disjointSet.Union(crd2Idx(1, 2), crd2Idx(0, 4));
  expectNodes(0, 1, 1, 1, 1, //
              0, 0, 1, 1, 1, //
              0, 0, 1, 1, 1, //
              0, 0, 0, 1, 1, //
              0, 0, 0, 0, 0);

  disjointSet.Union(crd2Idx(1, 4), crd2Idx(3, 1));
  expectNodes(0, 0, 0, 0, 0, //
              0, 0, 0, 0, 0, //
              0, 0, 0, 0, 0, //
              0, 0, 0, 0, 0, //
              0, 0, 0, 0, 0);
}

void TestCUDA() {
  using Volume = TensorVector<int, SpaceDevice>;

  // Initialize.
  Volume volume{make_layout_major(1000)};
  auto tensor = make_tensor(raw_pointer_cast(volume.tensor().data()), volume.layout());
  DisjointSet<ThreadSafe, decltype(tensor)> disjointSet(tensor);
  for (int x = 0; x < disjointSet.nodes().size(); ++x)
    volume(x) = x;

  // Parallel union.
  Launcher(volume.size() - 1, [=] ARIA_DEVICE(int x) mutable { disjointSet.Union(x + 1, x); }).Launch();

  Launcher(volume.size(), [=] ARIA_DEVICE(int x) mutable { disjointSet.FindAndCompress(x); }).Launch();

  cuda::device::current::get().synchronize();

  // Check
  Volume::Mirrored<SpaceHost> volumeH{volume.layout()};
  copy(volumeH, volume);

  cuda::device::current::get().synchronize();

  for (int x = 0; x < volumeH.size(); ++x) {
    EXPECT_EQ(volumeH(x), 0);
  }
}

} // namespace

TEST(DisjointSet, Base) {
  TestCPU<ThreadUnsafe, SpaceHost>();
  TestCPU<ThreadUnsafe, SpaceDevice>();
  TestCPU<ThreadSafe, SpaceHost>();
  // TestCPU<ThreadSafe, SpaceDevice>(); // Should not compile.

  TestCUDA();
}

} // namespace ARIA
