#include "ARIA/detail/MosaicIterator.h"
#include "ARIA/Let.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct Pattern {
  int i;
  float f;
};

} // namespace

template <>
struct Mosaic<Tup<int, float>, Pattern> {
  Pattern operator()(const Tup<int, float> &v) const { return {.i = get<0>(v), .f = get<1>(v)}; }

  Tup<int, float> operator()(const Pattern &v) const { return {v.i, v.f}; }
};

TEST(MosaicIterator, Mosaic) {
  using T = Tup<int, float>;
  using TMosaic = Mosaic<T, Pattern>;

  std::array<int, 5> is = {0, 1, 2, 3, 4};
  std::array<float, 5> fs = {0.1F, 1.2F, 2.3F, 3.4F, 4.5F};

  {
    let begin = make_mosaic_iterator<TMosaic>(Tup{is.begin(), fs.begin()});
    let end = make_mosaic_iterator<TMosaic>(Tup{is.end(), fs.end()});
    let beginC = make_mosaic_iterator<TMosaic>(Tup{is.cbegin(), fs.cbegin()});
    let endC = make_mosaic_iterator<TMosaic>(Tup{is.cend(), fs.cend()});

    static_assert(Property<decltype(*begin)>);
    static_assert(Property<decltype(*end)>);
    static_assert(Property<decltype(*beginC)>);
    static_assert(Property<decltype(*endC)>);
  }

  {
    let begin = make_mosaic_iterator<TMosaic>(Tup{is.begin(), fs.begin()});
    let end = make_mosaic_iterator<TMosaic>(Tup{is.end(), fs.end()});

    for (let it = begin; it != end; ++it) {
      let v = Let(*it);
      static_assert(std::is_same_v<decltype(v), T>);

      if (it == begin + 0) {
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_FLOAT_EQ(get<1>(v), 0.1F);
      } else if (it == begin + 1) {
        EXPECT_EQ(get<0>(v), 1);
        EXPECT_FLOAT_EQ(get<1>(v), 1.2F);
      } else if (it == begin + 2) {
        EXPECT_EQ(get<0>(v), 2);
        EXPECT_FLOAT_EQ(get<1>(v), 2.3F);
      } else if (it == begin + 3) {
        EXPECT_EQ(get<0>(v), 3);
        EXPECT_FLOAT_EQ(get<1>(v), 3.4F);
      } else if (it == begin + 4) {
        EXPECT_EQ(get<0>(v), 4);
        EXPECT_FLOAT_EQ(get<1>(v), 4.5F);
      }

      *it += T{10, 10.01F};
      v = *it;

      if (it == begin + 0) {
        EXPECT_EQ(get<0>(v), 10);
        EXPECT_FLOAT_EQ(get<1>(v), 10.11F);
      } else if (it == begin + 1) {
        EXPECT_EQ(get<0>(v), 11);
        EXPECT_FLOAT_EQ(get<1>(v), 11.21F);
      } else if (it == begin + 2) {
        EXPECT_EQ(get<0>(v), 12);
        EXPECT_FLOAT_EQ(get<1>(v), 12.31F);
      } else if (it == begin + 3) {
        EXPECT_EQ(get<0>(v), 13);
        EXPECT_FLOAT_EQ(get<1>(v), 13.41F);
      } else if (it == begin + 4) {
        EXPECT_EQ(get<0>(v), 14);
        EXPECT_FLOAT_EQ(get<1>(v), 14.51F);
      }
    }
  }
}

TEST(MosaicIterator, NonMosaic) {
  std::array<int, 5> is = {0, 1, 2, 3, 4};

  {
    let begin = make_mosaic_iterator<int>(is.begin());
    let end = make_mosaic_iterator<int>(is.end());
    let beginC = make_mosaic_iterator<int>(is.cbegin());
    let endC = make_mosaic_iterator<int>(is.cend());

    static_assert(!Property<decltype(*begin)>);
    static_assert(!Property<decltype(*end)>);
    static_assert(!Property<decltype(*beginC)>);
    static_assert(!Property<decltype(*endC)>);

    static_assert(std::is_same_v<decltype(begin), decltype(is.begin())>);
    static_assert(std::is_same_v<decltype(end), decltype(is.end())>);
    static_assert(std::is_same_v<decltype(beginC), decltype(is.cbegin())>);
    static_assert(std::is_same_v<decltype(endC), decltype(is.cend())>);

    static_assert(std::is_same_v<decltype(*begin), int &>);
    static_assert(std::is_same_v<decltype(*end), int &>);
    static_assert(std::is_same_v<decltype(*beginC), const int &>);
    static_assert(std::is_same_v<decltype(*endC), const int &>);
  }

  {
    let begin = make_mosaic_iterator<int>(is.begin());
    let end = make_mosaic_iterator<int>(is.end());

    for (let it = begin; it != end; ++it) {
      let v = *it;
      static_assert(std::is_same_v<decltype(v), int>);

      if (it == begin + 0) {
        EXPECT_EQ(v, 0);
      } else if (it == begin + 1) {
        EXPECT_EQ(v, 1);
      } else if (it == begin + 2) {
        EXPECT_EQ(v, 2);
      } else if (it == begin + 3) {
        EXPECT_EQ(v, 3);
      } else if (it == begin + 4) {
        EXPECT_EQ(v, 4);
      }

      *it += 10;
      v = *it;

      if (it == begin + 0) {
        EXPECT_EQ(v, 10);
      } else if (it == begin + 1) {
        EXPECT_EQ(v, 11);
      } else if (it == begin + 2) {
        EXPECT_EQ(v, 12);
      } else if (it == begin + 3) {
        EXPECT_EQ(v, 13);
      } else if (it == begin + 4) {
        EXPECT_EQ(v, 14);
      }
    }
  }
}

} // namespace ARIA
