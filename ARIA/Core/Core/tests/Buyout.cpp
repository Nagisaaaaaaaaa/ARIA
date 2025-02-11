#include "ARIA/Buyout.h"
#include "ARIA/Let.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct SizeOf {
  template <typename T>
  constexpr size_t operator()() const {
    return sizeof(T);
  }
};

struct Construct {
  template <typename T>
  constexpr T operator()() const {
    if constexpr (std::integral<T>)
      return T(9);
    else
      return T{9};
  }
};

struct ConstructNonConstexpr {
  template <typename T>
  T operator()() const {
    if constexpr (std::integral<T>)
      return T(9);
    else
      return T{9};
  }
};

} // namespace

TEST(Buyout, Base) {
  {
    using TBuyout0 = const Buyout<SizeOf, int8_t, int16_t, int32_t>;
    using TBuyout1 = const Buyout<SizeOf, MakeTypeArray<int8_t, int16_t, int32_t>>;
    using TBuyout2 = const Buyout<SizeOf, MakeTypeSet<int8_t, int16_t, int32_t>>;
    constexpr let buyout0 = make_buyout<int8_t, int16_t, int32_t>(SizeOf{});
    constexpr let buyout1 = make_buyout<MakeTypeArray<int8_t, int16_t, int32_t>>(SizeOf{});
    constexpr let buyout2 = make_buyout<MakeTypeSet<int8_t, int16_t, int32_t>>(SizeOf{});

    static_assert(std::is_same_v<TBuyout1, TBuyout0>);
    static_assert(std::is_same_v<TBuyout2, TBuyout0>);
    static_assert(std::is_same_v<decltype(buyout0), TBuyout0>);
    static_assert(std::is_same_v<decltype(buyout1), TBuyout0>);
    static_assert(std::is_same_v<decltype(buyout2), TBuyout0>);
  }
}

TEST(Buyout, Get) {
  {
    constexpr let buyout = make_buyout<int8_t, int16_t, int32_t, int64_t>(SizeOf{});
    static_assert(std::is_same_v<decltype(buyout), const Buyout<SizeOf, int8_t, int16_t, int32_t, int64_t>>);

    static_assert(buyout.operator()<int8_t>() == 1);
    static_assert(buyout.operator()<int16_t>() == 2);
    static_assert(buyout.operator()<int32_t>() == 4);
    static_assert(buyout.operator()<int64_t>() == 8);

    static_assert(get<int8_t>(buyout) == 1);
    static_assert(get<int16_t>(buyout) == 2);
    static_assert(get<int32_t>(buyout) == 4);
    static_assert(get<int64_t>(buyout) == 8);
  }

  {
    constexpr let buyout = make_buyout<int8_t, int16_t, int32_t, int64_t, std::array<int, 1>>(Construct{});
    static_assert(std::is_same_v<decltype(buyout),
                                 const Buyout<Construct, int8_t, int16_t, int32_t, int64_t, std::array<int, 1>>>);

    static_assert(std::is_same_v<decltype(buyout.operator()<int8_t>()), const int8_t &>);
    static_assert(std::is_same_v<decltype(buyout.operator()<int16_t>()), const int16_t &>);
    static_assert(std::is_same_v<decltype(buyout.operator()<int32_t>()), const int32_t &>);
    static_assert(std::is_same_v<decltype(buyout.operator()<int64_t>()), const int64_t &>);
    static_assert(std::is_same_v<decltype(buyout.operator()<std::array<int, 1>>()), const std::array<int, 1> &>);

    static_assert(std::is_same_v<decltype(get<int8_t>(buyout)), const int8_t &>);
    static_assert(std::is_same_v<decltype(get<int16_t>(buyout)), const int16_t &>);
    static_assert(std::is_same_v<decltype(get<int32_t>(buyout)), const int32_t &>);
    static_assert(std::is_same_v<decltype(get<int64_t>(buyout)), const int64_t &>);
    static_assert(std::is_same_v<decltype(get<std::array<int, 1>>(buyout)), const std::array<int, 1> &>);

    static_assert(buyout.operator()<int8_t>() == 9);
    static_assert(buyout.operator()<int16_t>() == 9);
    static_assert(buyout.operator()<int32_t>() == 9);
    static_assert(buyout.operator()<int64_t>() == 9);
    static_assert(buyout.operator()<std::array<int, 1>>() == std::array{9});

    static_assert(get<int8_t>(buyout) == 9);
    static_assert(get<int16_t>(buyout) == 9);
    static_assert(get<int32_t>(buyout) == 9);
    static_assert(get<int64_t>(buyout) == 9);
    static_assert(get<std::array<int, 1>>(buyout) == std::array{9});
  }

  {
    let buyout = make_buyout<int8_t, int16_t, int32_t, int64_t, std::array<int, 1>>(ConstructNonConstexpr{});
    static_assert(std::is_same_v<decltype(buyout),
                                 Buyout<ConstructNonConstexpr, int8_t, int16_t, int32_t, int64_t, std::array<int, 1>>>);

    static_assert(std::is_same_v<decltype(buyout.operator()<int8_t>()), const int8_t &>);
    static_assert(std::is_same_v<decltype(buyout.operator()<int16_t>()), const int16_t &>);
    static_assert(std::is_same_v<decltype(buyout.operator()<int32_t>()), const int32_t &>);
    static_assert(std::is_same_v<decltype(buyout.operator()<int64_t>()), const int64_t &>);
    static_assert(std::is_same_v<decltype(buyout.operator()<std::array<int, 1>>()), const std::array<int, 1> &>);

    static_assert(std::is_same_v<decltype(get<int8_t>(buyout)), const int8_t &>);
    static_assert(std::is_same_v<decltype(get<int16_t>(buyout)), const int16_t &>);
    static_assert(std::is_same_v<decltype(get<int32_t>(buyout)), const int32_t &>);
    static_assert(std::is_same_v<decltype(get<int64_t>(buyout)), const int64_t &>);
    static_assert(std::is_same_v<decltype(get<std::array<int, 1>>(buyout)), const std::array<int, 1> &>);

    EXPECT_EQ(buyout.operator()<int8_t>(), 9);
    EXPECT_EQ(buyout.operator()<int16_t>(), 9);
    EXPECT_EQ(buyout.operator()<int32_t>(), 9);
    EXPECT_EQ(buyout.operator()<int64_t>(), 9);
    {
      let v = buyout.operator()<std::array<int, 1>>();
      EXPECT_EQ(v, std::array{9});
    }

    EXPECT_EQ(get<int8_t>(buyout), 9);
    EXPECT_EQ(get<int16_t>(buyout), 9);
    EXPECT_EQ(get<int32_t>(buyout), 9);
    EXPECT_EQ(get<int64_t>(buyout), 9);
    {
      let v = get<std::array<int, 1>>(buyout);
      EXPECT_EQ(v, std::array{9});
    }
  }
}

} // namespace ARIA
