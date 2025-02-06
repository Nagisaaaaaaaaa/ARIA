#include "ARIA/Let.h"
#include "ARIA/detail/MosaicArray.h"

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
struct Mosaic<Tec<int, int, int>, PatternIII> {
  PatternIII operator()(const Tec<int, int, int> &v) const {
    return {.ii = {.v0 = get<0>(v), .v1 = get<1>(v)}, .v2 = get<2>(v)};
  }

  Tec<int, int, int> operator()(const PatternIII &v) const { return {v.ii.v0, v.ii.v1, v.v2}; }
};

TEST(MosaicArray, Base) {
  using namespace mosaic::detail;

  // `int, float`.
  {
    using T = Tup<int, float>;
    using TMosaic = Mosaic<T, PatternIF>;

    using TMosaicArray = MosaicArray<TMosaic, 5>;

    TMosaicArray vec{};
    static_assert(vec.size() == 5);

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
        v = vProp;
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_FLOAT_EQ(get<1>(v), 0.0F);

        ++i;
      }
    }

    {
      TMosaicArray vec1{{T{0, 0.1F}, T{1, 1.2F}, T{2, 2.3F}, T{3, 3.4F}, T{4, 4.5F}}};
      for (int i = 0; i < 5; ++i) {
        T v = vec1[i];
        EXPECT_EQ(get<0>(v), i);
        EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);
      }

      vec1.fill(T{0, 0.0F});
      for (int i = 0; i < 5; ++i) {
        T v = vec1[i];
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_FLOAT_EQ(get<1>(v), 0.0F);
      }
    }
  }

  // `int, int, int`.
  {
    using T = Tec<int, int, int>;
    using TMosaic = Mosaic<T, PatternIII>;

    using TMosaicArray = MosaicArray<TMosaic, 5>;

    TMosaicArray vec{};
    static_assert(vec.size() == 5);

    for (int i = 0; i < 5; ++i) {
      let v = Let(vec[i]);
      static_assert(Property<decltype(vec[i])>);
      static_assert(std::is_same_v<decltype(v), T>);
      EXPECT_EQ(get<0>(v), 0);
      EXPECT_EQ(get<1>(v), 0);
      EXPECT_EQ(get<2>(v), 0);

      vec[i] = {i, 2 * i, 3 * i};
      v = vec[i];
      EXPECT_EQ(get<0>(v), i);
      EXPECT_EQ(get<1>(v), 2 * i);
      EXPECT_EQ(get<2>(v), 3 * i);
    }

    for (let it = vec.begin(); it != vec.end(); ++it) {
      let v = Let(*it);
      let k = it - vec.begin();
      int i = k;
      static_assert(Property<decltype(*it)>);
      static_assert(std::is_same_v<decltype(v), T>);
      static_assert(std::is_same_v<decltype(k), int64>);
      EXPECT_EQ(get<0>(v), i);
      EXPECT_EQ(get<1>(v), 2 * i);
      EXPECT_EQ(get<2>(v), 3 * i);

      *it -= T{i, 2 * i, 3 * i};
      v = vec[i];
      EXPECT_EQ(get<0>(v), 0);
      EXPECT_EQ(get<1>(v), 0);
      EXPECT_EQ(get<2>(v), 0);
    }

    for (let it = vec.cbegin(); it != vec.cend(); ++it) {
      let v = Let(*it);
      let k = it - vec.cbegin();
      static_assert(Property<decltype(*it)>);
      static_assert(std::is_same_v<decltype(v), T>);
      static_assert(std::is_same_v<decltype(k), int64>);
      EXPECT_EQ(get<0>(v), 0);
      EXPECT_EQ(get<1>(v), 0);
      EXPECT_EQ(get<2>(v), 0);
    }

    for (int i = 0; i < 5; ++i) {
      let ptr = vec.data() + i;
      let v = Let(*ptr);
      static_assert(Property<decltype(*ptr)>);
      static_assert(std::is_same_v<decltype(v), T>);
      EXPECT_EQ(get<0>(v), 0);
      EXPECT_EQ(get<1>(v), 0);
      EXPECT_EQ(get<2>(v), 0);

      *ptr += T{i, 2 * i, 3 * i};
      v = *ptr;
      EXPECT_EQ(get<0>(v), i);
      EXPECT_EQ(get<1>(v), 2 * i);
      EXPECT_EQ(get<2>(v), 3 * i);
    }

    {
      int i = 0;
      for (auto vProp : vec) {
        let v = Let(vProp);
        static_assert(Property<decltype(vProp)>);
        static_assert(std::is_same_v<decltype(v), T>);
        EXPECT_EQ(get<0>(v), i);
        EXPECT_EQ(get<1>(v), 2 * i);
        EXPECT_EQ(get<2>(v), 3 * i);

        vProp *= 0;
        v = vProp;
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_EQ(get<1>(v), 0);
        EXPECT_EQ(get<2>(v), 0);

        ++i;
      }
    }

    {
      TMosaicArray vec1{{T{0, 0, 0}, T{1, 2, 3}, T{2, 4, 6}, T{3, 6, 9}, T{4, 8, 12}}};
      for (int i = 0; i < 5; ++i) {
        T v = vec1[i];
        EXPECT_EQ(get<0>(v), i);
        EXPECT_EQ(get<1>(v), 2 * i);
        EXPECT_EQ(get<2>(v), 3 * i);
      }

      vec1.fill(T{0, 0, 0});
      for (int i = 0; i < 5; ++i) {
        T v = vec1[i];
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_EQ(get<1>(v), 0);
        EXPECT_EQ(get<2>(v), 0);
      }
    }
  }
}

TEST(MosaicArray, Copy) {
  using namespace mosaic::detail;

  // `int, float`.
  {
    using T = Tup<int, float>;
    using TMosaic = Mosaic<T, PatternIF>;

    using TMosaicArray = MosaicArray<TMosaic, 5>;
    using TArray = cuda::std::array<T, 5>;

    TArray vec0{T{0, 0.1F}, T{1, 1.2F}, T{2, 2.3F}, T{3, 3.4F}, T{4, 4.5F}};
    TMosaicArray vec1 = vec0;
    for (int i = 0; i < 5; ++i) {
      T v = vec1[i];
      EXPECT_EQ(get<0>(v), i);
      EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);
    }
  }

  // `int, int, int`.
  {
    using T = Tec<int, int, int>;
    using TMosaic = Mosaic<T, PatternIII>;

    using TMosaicArray = MosaicArray<TMosaic, 5>;
    using TArray = cuda::std::array<T, 5>;

    TArray vec0{T{0, 0, 0}, T{1, 2, 3}, T{2, 4, 6}, T{3, 6, 9}, T{4, 8, 12}};
    TMosaicArray vec1 = vec0;
    for (int i = 0; i < 5; ++i) {
      T v = vec1[i];
      EXPECT_EQ(get<0>(v), i);
      EXPECT_EQ(get<1>(v), 2 * i);
      EXPECT_EQ(get<2>(v), 3 * i);
    }
  }
}

} // namespace ARIA
