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
  for (int y = 0; y < disjointSet.labels().size<1>(); ++y)
    for (int x = 0; x < disjointSet.labels().size<0>(); ++x)
      disjointSet.labels()(x, y) = disjointSet.labels().layout()(x, y);

  // Supporting functions.
  auto crd2Idx = [&](auto x, auto y) { return flags.layout()(x, y); };
#if 0
  auto printLabels = [&]() {
    for (int y = 0; y < disjointSet.labels().size<1>(); ++y) {
      for (int x = 0; x < disjointSet.labels().size<0>(); ++x) {
        fmt::print("{} ", disjointSet.labels()(x, y));
      }
      fmt::print("\n");
    }
  };
#endif
  auto expectLabel = [&](auto x, auto y, auto label) {
    EXPECT_EQ(disjointSet.FindAndCompress(crd2Idx(x, y)), label);
    EXPECT_EQ(disjointSet.Find(crd2Idx(x, y)), label);
    EXPECT_EQ(disjointSet.labels()(x, y), label);
  };
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

  // Simple union.
  expectLabels(0, 1, 2, 3, 4,      //
               5, 6, 7, 8, 9,      //
               10, 11, 12, 13, 14, //
               15, 16, 17, 18, 19, //
               20, 21, 22, 23, 24);

  disjointSet.Union(crd2Idx(0, 4), crd2Idx(0, 4));
  expectLabels(0, 1, 2, 3, 4,      //
               5, 6, 7, 8, 9,      //
               10, 11, 12, 13, 14, //
               15, 16, 17, 18, 19, //
               20, 21, 22, 23, 24);

  disjointSet.Union(crd2Idx(0, 4), crd2Idx(1, 4));
  expectLabels(0, 1, 2, 3, 4,      //
               5, 6, 7, 8, 9,      //
               10, 11, 12, 13, 14, //
               15, 16, 17, 18, 19, //
               20, 20, 22, 23, 24);

  disjointSet.Union(crd2Idx(0, 3), crd2Idx(0, 2));
  expectLabels(0, 1, 2, 3, 4,      //
               5, 6, 7, 8, 9,      //
               10, 11, 12, 13, 14, //
               10, 16, 17, 18, 19, //
               20, 20, 22, 23, 24);

  disjointSet.Union(crd2Idx(1, 4), crd2Idx(0, 2));
  expectLabels(0, 1, 2, 3, 4,      //
               5, 6, 7, 8, 9,      //
               10, 11, 12, 13, 14, //
               10, 16, 17, 18, 19, //
               10, 10, 22, 23, 24);

  disjointSet.Union(crd2Idx(1, 4), crd2Idx(1, 4));
  expectLabels(0, 1, 2, 3, 4,      //
               5, 6, 7, 8, 9,      //
               10, 11, 12, 13, 14, //
               10, 16, 17, 18, 19, //
               10, 10, 22, 23, 24);

  // Complex union.
  for (int y = 0; y < disjointSet.labels().size<1>(); ++y)
    for (int x = 0; x < disjointSet.labels().size<0>(); ++x)
      if (flags(x, y) == L) {
        if (x - 1 >= 0 && flags(x - 1, y) == L)
          disjointSet.Union(crd2Idx(x - 1, y), crd2Idx(x, y));
        if (x + 1 < disjointSet.labels().size<0>() && flags(x + 1, y) == L)
          disjointSet.Union(crd2Idx(x + 1, y), crd2Idx(x, y));
        if (y - 1 >= 0 && flags(x, y - 1) == L)
          disjointSet.Union(crd2Idx(x, y - 1), crd2Idx(x, y));
        if (y + 1 < disjointSet.labels().size<1>() && flags(x, y + 1) == L)
          disjointSet.Union(crd2Idx(x, y + 1), crd2Idx(x, y));
      }
  expectLabels(0, 1, 1, 1, 1,    //
               5, 6, 1, 1, 1,    //
               10, 11, 1, 1, 1,  //
               10, 16, 17, 1, 1, //
               10, 10, 22, 23, 24);

  for (int y = 0; y < disjointSet.labels().size<1>(); ++y)
    for (int x = 0; x < disjointSet.labels().size<0>(); ++x)
      if (flags(x, y) == I) {
        if (x - 1 >= 0 && flags(x - 1, y) == I)
          disjointSet.Union(crd2Idx(x - 1, y), crd2Idx(x, y));
        if (x + 1 < disjointSet.labels().size<0>() && flags(x + 1, y) == I)
          disjointSet.Union(crd2Idx(x + 1, y), crd2Idx(x, y));
        if (y - 1 >= 0 && flags(x, y - 1) == I)
          disjointSet.Union(crd2Idx(x, y - 1), crd2Idx(x, y));
        if (y + 1 < disjointSet.labels().size<1>() && flags(x, y + 1) == I)
          disjointSet.Union(crd2Idx(x, y + 1), crd2Idx(x, y));
      }
  expectLabels(0, 1, 1, 1, 1,  //
               0, 0, 1, 1, 1,  //
               10, 0, 1, 1, 1, //
               10, 0, 0, 1, 1, //
               10, 10, 0, 0, 0);

  disjointSet.Union(crd2Idx(1, 2), crd2Idx(0, 4));
  expectLabels(0, 1, 1, 1, 1, //
               0, 0, 1, 1, 1, //
               0, 0, 1, 1, 1, //
               0, 0, 0, 1, 1, //
               0, 0, 0, 0, 0);

  disjointSet.Union(crd2Idx(1, 4), crd2Idx(3, 1));
  expectLabels(0, 0, 0, 0, 0, //
               0, 0, 0, 0, 0, //
               0, 0, 0, 0, 0, //
               0, 0, 0, 0, 0, //
               0, 0, 0, 0, 0);
}

void TestCUDA() {
  using Volume = TensorVector<int, SpaceDevice>;

  // Initialize.
  Volume volume{make_layout_major(1000)};
  auto tensor = cute::make_tensor(volume.tensor().data().get(), volume.layout());
  DisjointSet<ThreadSafe, decltype(tensor)> disjointSet(tensor);
  for (int x = 0; x < disjointSet.labels().size(); ++x)
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
