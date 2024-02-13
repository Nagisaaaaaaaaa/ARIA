#include "ARIA/ForEach.h"

#include <gtest/gtest.h>

#include <array>

namespace ARIA {

TEST(ForEach, Base) {
  std::vector<int> one1;
  for (int i = 0; i < 3; ++i)
    one1.push_back(i);

  std::vector<std::array<int, 3>> three1;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        three1.push_back({i, j, k});
      }

  // ForEach non-const parameter, lambda parameter.
  {
    std::vector<std::array<int, 3>> three0;

    ForEach(3, [&](auto i) {
      ForEach(3, [&](auto j) {
        ForEach(3, [&](auto k) {
          static_assert(std::is_same_v<decltype(i), int>);
          static_assert(std::is_same_v<decltype(j), int>);
          static_assert(std::is_same_v<decltype(k), int>);

          const auto x = i;
          const auto y = j;
          const auto z = k;

          three0.push_back({x, y, z});
        });
      });
    });

    EXPECT_EQ(three0, three1);
  }

  // ForEach const parameter, lambda parameter.
  {
    std::vector<int> one0;
    std::vector<std::array<int, 3>> three0;

    ForEach(C<3>{}, [&](auto i) {
      static_assert(ConstantIntegral<decltype(i)>);

      constexpr auto x = i;

      one0.push_back(x);
    });

    ForEach(C<3>{}, [&]<auto i>(C<i>) {
      ForEach(C<3>{}, [&]<auto j>(C<j>) {
        ForEach(C<3>{}, [&]<auto k>(C<k>) {
          static_assert(std::is_same_v<decltype(i), int>);
          static_assert(std::is_same_v<decltype(j), int>);
          static_assert(std::is_same_v<decltype(k), int>);

          constexpr auto x = i;
          constexpr auto y = j;
          constexpr auto z = k;

          three0.push_back({x, y, z});
        });
      });
    });

    EXPECT_EQ(one0, one1);
    EXPECT_EQ(three0, three1);
  }

  // ForEach const parameter, lambda t-parameter.
  {
    std::vector<std::array<int, 3>> three0;

    ForEach(C<3>{}, [&]<auto i> {
      ForEach(C<3>{}, [&]<auto j> {
        ForEach(C<3>{}, [&]<auto k> {
          static_assert(std::is_same_v<decltype(i), int>);
          static_assert(std::is_same_v<decltype(j), int>);
          static_assert(std::is_same_v<decltype(k), int>);

          constexpr auto x = i;
          constexpr auto y = j;
          constexpr auto z = k;

          three0.push_back({x, y, z});
        });
      });
    });

    EXPECT_EQ(three0, three1);
  }

  // ForEach const t-parameter, lambda parameter.
  {
    std::vector<int> one0;
    std::vector<std::array<int, 3>> three0;

    ForEach<3>([&](auto i) {
      static_assert(ConstantIntegral<decltype(i)>);

      constexpr auto x = i;

      one0.push_back(x);
    });

    ForEach<3>([&]<auto i>(C<i>) {
      ForEach<3>([&]<auto j>(C<j>) {
        ForEach<3>([&]<auto k>(C<k>) {
          static_assert(std::is_same_v<decltype(i), int>);
          static_assert(std::is_same_v<decltype(j), int>);
          static_assert(std::is_same_v<decltype(k), int>);

          constexpr auto x = i;
          constexpr auto y = j;
          constexpr auto z = k;

          three0.push_back({x, y, z});
        });
      });
    });

    EXPECT_EQ(one0, one1);
    EXPECT_EQ(three0, three1);
  }

  // ForEach const t-parameter, lambda t-parameter.
  {
    std::vector<std::array<int, 3>> three0;

    ForEach<3>([&]<auto i> {
      ForEach<3>([&]<auto j> {
        ForEach<3>([&]<auto k> {
          static_assert(std::is_same_v<decltype(i), int>);
          static_assert(std::is_same_v<decltype(j), int>);
          static_assert(std::is_same_v<decltype(k), int>);

          constexpr auto x = i;
          constexpr auto y = j;
          constexpr auto z = k;

          three0.push_back({x, y, z});
        });
      });
    });

    EXPECT_EQ(three0, three1);
  }

  // ForEach const t-parameter, lambda parameter.
  {
    std::vector<int> one0;
    std::vector<std::array<int, 3>> three0;

    ForEach<C<3>>([&](auto i) {
      static_assert(ConstantIntegral<decltype(i)>);

      constexpr auto x = i;

      one0.push_back(x);
    });

    ForEach<C<3>>([&]<auto i>(C<i>) {
      ForEach<C<3>>([&]<auto j>(C<j>) {
        ForEach<C<3>>([&]<auto k>(C<k>) {
          static_assert(std::is_same_v<decltype(i), int>);
          static_assert(std::is_same_v<decltype(j), int>);
          static_assert(std::is_same_v<decltype(k), int>);

          constexpr auto x = i;
          constexpr auto y = j;
          constexpr auto z = k;

          three0.push_back({x, y, z});
        });
      });
    });

    EXPECT_EQ(one0, one1);
    EXPECT_EQ(three0, three1);
  }

  // ForEach const t-parameter, lambda t-parameter.
  {
    std::vector<std::array<int, 3>> three0;

    ForEach<C<3>>([&]<auto i> {
      ForEach<C<3>>([&]<auto j> {
        ForEach<C<3>>([&]<auto k> {
          static_assert(std::is_same_v<decltype(i), int>);
          static_assert(std::is_same_v<decltype(j), int>);
          static_assert(std::is_same_v<decltype(k), int>);

          constexpr auto x = i;
          constexpr auto y = j;
          constexpr auto z = k;

          three0.push_back({x, y, z});
        });
      });
    });

    EXPECT_EQ(three0, three1);
  }
}

} // namespace ARIA
