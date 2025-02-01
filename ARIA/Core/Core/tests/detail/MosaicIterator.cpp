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

    static_assert(std::is_same_v<decltype(*begin), Pattern>);
    static_assert(std::is_same_v<decltype(*end), Pattern>);
    static_assert(std::is_same_v<decltype(*beginC), Pattern>);
    static_assert(std::is_same_v<decltype(*endC), Pattern>);
  }

  {
    let begin = make_mosaic_iterator<TMosaic>(Tup{is.begin(), fs.begin()});
    let end = make_mosaic_iterator<TMosaic>(Tup{is.end(), fs.end()});

    for (let it = begin; it != end; ++it) {
      if (it == begin + 0) {
        EXPECT_EQ(it->i, 0);
        EXPECT_FLOAT_EQ(it->f, 0.1F);
      } else if (it == begin + 1) {
        EXPECT_EQ(it->i, 1);
        EXPECT_FLOAT_EQ(it->f, 1.2F);
      } else if (it == begin + 2) {
        EXPECT_EQ(it->i, 2);
        EXPECT_FLOAT_EQ(it->f, 2.3F);
      } else if (it == begin + 3) {
        EXPECT_EQ(it->i, 3);
        EXPECT_FLOAT_EQ(it->f, 3.4F);
      } else if (it == begin + 4) {
        EXPECT_EQ(it->i, 4);
        EXPECT_FLOAT_EQ(it->f, 4.5F);
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

    static_assert(std::is_same_v<decltype(begin), decltype(is.begin())>);
    static_assert(std::is_same_v<decltype(end), decltype(is.end())>);
    static_assert(std::is_same_v<decltype(beginC), decltype(is.cbegin())>);
    static_assert(std::is_same_v<decltype(endC), decltype(is.cend())>);
  }

  {
    let begin = make_mosaic_iterator<int>(is.begin());
    let end = make_mosaic_iterator<int>(is.end());

    for (let it = begin; it != end; ++it) {
      if (it == begin + 0) {
        EXPECT_EQ(*it, 0);
      } else if (it == begin + 1) {
        EXPECT_EQ(*it, 1);
      } else if (it == begin + 2) {
        EXPECT_EQ(*it, 2);
      } else if (it == begin + 3) {
        EXPECT_EQ(*it, 3);
      } else if (it == begin + 4) {
        EXPECT_EQ(*it, 4);
      }
    }
  }
}

} // namespace ARIA
