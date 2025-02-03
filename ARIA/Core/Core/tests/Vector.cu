#include "ARIA/Let.h"
#include "ARIA/Vector.h"

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

TEST(Vector, Base) {
  // Mosaic.
  {
    ForEach<MakeTypeArray<                     //
        Mosaic<Tup<int, float>, PatternIF>,    //
        Mosaic<Tup<int, int, int>, PatternIII> //
        >>([]<typename TMosaic>() {
      static_assert(std::is_same_v<Vector<TMosaic, SpaceHost>, mosaic::detail::MosaicVector<TMosaic, SpaceHost>>);
      static_assert(std::is_same_v<VectorHost<TMosaic>, mosaic::detail::MosaicVector<TMosaic, SpaceHost>>);

      static_assert(std::is_same_v<Vector<TMosaic, SpaceDevice>, mosaic::detail::MosaicVector<TMosaic, SpaceDevice>>);
      static_assert(std::is_same_v<VectorDevice<TMosaic>, mosaic::detail::MosaicVector<TMosaic, SpaceDevice>>);
    });
  }

  // Non-mosaic.
  {
    ForEach<MakeTypeArray<                          //
        int,                                        //
        Tup<int, float, double>,                    //
        Tup<Tec2i<int, Int<5>>, Tec<float, double>> //
        >>([]<typename T>() {
      static_assert(std::is_same_v<Vector<T, SpaceHost>, thrust::host_vector<T>>);
      static_assert(std::is_same_v<VectorHost<T>, thrust::host_vector<T>>);

      static_assert(std::is_same_v<Vector<T, SpaceDevice>, thrust::device_vector<T>>);
      static_assert(std::is_same_v<VectorDevice<T>, thrust::device_vector<T>>);
    });
  }

  // `int, float`.
  {
    using T = Tup<int, float>;
    using TMosaic = Mosaic<T, PatternIF>;

    ForEach<MakeTypeArray<     //
        VectorHost<TMosaic>,   //
        VectorDevice<TMosaic>, //
        VectorHost<T>,         //
        VectorDevice<T>        //
        >>([]<typename TVector>() {
      TVector vec;
      EXPECT_EQ(vec.size(), 0);

      vec.resize(5);
      EXPECT_EQ(vec.size(), 5);

      for (int i = 0; i < 5; ++i) {
        T v = vec[i];
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_FLOAT_EQ(get<1>(v), 0.0F);

        vec[i] = T{i, i + (i + 1) * 0.1F};
        v = vec[i];
        EXPECT_EQ(get<0>(v), i);
        EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);
      }

      for (let it = vec.begin(); it != vec.end(); ++it) {
        T v = *it;
        let k = it - vec.begin();
        int i = k;
        static_assert(std::is_same_v<decltype(k), int64>);
        EXPECT_EQ(get<0>(v), i);
        EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);

        *it = T{0, 0.0F};
        v = vec[i];
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_FLOAT_EQ(get<1>(v), 0.0F);
      }

      for (let it = vec.cbegin(); it != vec.cend(); ++it) {
        T v = *it;
        let k = it - vec.cbegin();
        static_assert(std::is_same_v<decltype(k), int64>);
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_FLOAT_EQ(get<1>(v), 0.0F);
      }

      for (int i = 0; i < 5; ++i) {
        let ptr = vec.data() + i;
        T v = *ptr;
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_FLOAT_EQ(get<1>(v), 0.0F);

        *ptr = T{i, i + (i + 1) * 0.1F};
        v = *ptr;
        EXPECT_EQ(get<0>(v), i);
        EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);
      }

      {
        TVector vec1(5);
        EXPECT_EQ(vec1.size(), 5);
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(get<0>(v), 0);
          EXPECT_FLOAT_EQ(get<1>(v), 0.0F);
        }

        vec1.clear();
        EXPECT_EQ(vec1.size(), 0);
      }

      {
        TVector vec1{T{0, 0.1F}, T{1, 1.2F}, T{2, 2.3F}, T{3, 3.4F}, T{4, 4.5F}};
        EXPECT_EQ(vec1.size(), 5);
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(get<0>(v), i);
          EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);
        }

        vec1.clear();
        EXPECT_EQ(vec1.size(), 0);
      }
    });
  }
}

} // namespace ARIA
