#include "ARIA/Let.h"
#include "ARIA/detail/MosaicVector.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct PatternIF {
  int i;
  float f;
};

struct PatternIII {
  struct {
    int v0;
    int v1;
  } ii;

  int v2;
};

} // namespace

template <>
struct Mosaic<Tup<int, float>, PatternIF> {
  PatternIF operator()(const Tup<int, float> &v) const { return {.i = get<0>(v), .f = get<1>(v)}; }

  Tup<int, float> operator()(const PatternIF &v) const { return {v.i, v.f}; }
};

template <>
struct Mosaic<Tup<int, int, int>, PatternIII> {
  PatternIII operator()(const Tup<int, int, int> &v) const {
    return {.ii = {.v0 = get<0>(v), .v1 = get<1>(v)}, .v2 = get<2>(v)};
  }

  Tup<int, int, int> operator()(const PatternIII &v) const { return {v.ii.v0, v.ii.v1, v.v2}; }
};

TEST(MosaicVector, Base) {
  using namespace mosaic::detail;

  // `int, float`.
  {
    using T = Tup<int, float>;
    using TMosaic = Mosaic<T, PatternIF>;

    MosaicVector<TMosaic> vec;
    EXPECT_EQ(vec.size(), 0);

    {
      MosaicVector<TMosaic> vec1(5);
      EXPECT_EQ(vec1.size(), 5);
    }

    vec.resize(5);
    EXPECT_EQ(vec.size(), 5);

    for (int i = 0; i < 5; ++i) {
      let v = Let(vec[i]);
      static_assert(Property<decltype(vec[i])>);
      static_assert(std::is_same_v<decltype(v), T>);
      EXPECT_EQ(get<0>(v), 0);
      EXPECT_FLOAT_EQ(get<1>(v), 0.0F);

      vec[i] = T{i, i + (i + 1) * 0.1F};
      v = vec[i];
      EXPECT_EQ(get<0>(v), i);
      EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);
    }

    for (let it = vec.begin(); it != vec.end(); ++it) {
      let v = Let(*it);
      let k = it - vec.begin();
      int i = k;
      static_assert(Property<decltype(*it)>);
      static_assert(std::is_same_v<decltype(v), T>);
      static_assert(std::is_same_v<decltype(k), int64>);
      EXPECT_EQ(get<0>(v), i);
      EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);

      *it -= T{i, i + (i + 1) * 0.1F};
      v = vec[i];
      EXPECT_EQ(get<0>(v), 0);
      EXPECT_FLOAT_EQ(get<1>(v), 0.0F);
    }

    for (let it = vec.cbegin(); it != vec.cend(); ++it) {
      let v = Let(*it);
      let k = it - vec.cbegin();
      int i = k;
      static_assert(Property<decltype(*it)>);
      static_assert(std::is_same_v<decltype(v), T>);
      static_assert(std::is_same_v<decltype(k), int64>);
      EXPECT_EQ(get<0>(v), 0);
      EXPECT_FLOAT_EQ(get<1>(v), 0.0F);
    }

    for (int i = 0; i < 5; ++i) {
      let ptr = vec.data() + i;
      let v = Let(*ptr);
      static_assert(Property<decltype(*ptr)>);
      static_assert(std::is_same_v<decltype(v), T>);
      EXPECT_EQ(get<0>(v), 0);
      EXPECT_FLOAT_EQ(get<1>(v), 0.0F);

      *ptr += T{i, i + (i + 1) * 0.1F};
      v = *ptr;
      EXPECT_EQ(get<0>(v), i);
      EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);
    }

    {
      int i = 0;
      for (auto vProp : vec) {
        let v = Let(vProp);
        static_assert(Property<decltype(vProp)>);
        static_assert(std::is_same_v<decltype(v), T>);
        EXPECT_EQ(get<0>(v), i);
        EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);

        vProp *= 0;
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_FLOAT_EQ(get<1>(v), 0.0F);

        ++i;
      }
    }
  }
}

} // namespace ARIA
