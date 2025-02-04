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

template <typename T>
class Vec3 {
public:
  Vec3() = default;

  ARIA_HOST_DEVICE Vec3(const T &x, const T &y, const T &z) : x_(x), y_(y), z_(z) {}

  ARIA_COPY_MOVE_ABILITY(Vec3, default, default);

  ARIA_REF_PROP(public, , x, x_);
  ARIA_REF_PROP(public, , y, y_);
  ARIA_REF_PROP(public, , z, z_);

private:
  T x_{}, y_{}, z_{};
};

template <typename T>
struct PatternVec3 {
  T x, y, z;
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

template <typename T>
struct Mosaic<Vec3<T>, PatternVec3<T>> {
  PatternVec3<T> operator()(const Vec3<T> &v) const { return {.x = v.x(), .y = v.y(), .z = v.z()}; }

  Vec3<T> operator()(const PatternVec3<T> &v) const { return {v.x, v.y, v.z}; }
};

TEST(Vector, Base) {
  // Mosaic.
  {
    ForEach<MakeTypeArray<                      //
        Mosaic<Tup<int, float>, PatternIF>,     //
        Mosaic<Tup<int, int, int>, PatternIII>, //
        Mosaic<Vec3<int>, PatternVec3<int>>,    //
        Mosaic<Vec3<float>, PatternVec3<float>> //
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
}

TEST(Vector, Methods) {
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

  // `int, int, int`.
  {
    using T = Tec<int, int, int>;
    using TMosaic = Mosaic<T, PatternIII>;

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
        EXPECT_EQ(get<1>(v), 0);
        EXPECT_EQ(get<2>(v), 0);

        vec[i] = T{i, 2 * i, 3 * i};
        v = vec[i];
        EXPECT_EQ(get<0>(v), i);
        EXPECT_EQ(get<1>(v), 2 * i);
        EXPECT_EQ(get<2>(v), 3 * i);
      }

      for (let it = vec.begin(); it != vec.end(); ++it) {
        T v = *it;
        let k = it - vec.begin();
        int i = k;
        static_assert(std::is_same_v<decltype(k), int64>);
        EXPECT_EQ(get<0>(v), i);
        EXPECT_EQ(get<1>(v), 2 * i);
        EXPECT_EQ(get<2>(v), 3 * i);

        *it = T{0, 0, 0};
        v = vec[i];
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_EQ(get<1>(v), 0);
        EXPECT_EQ(get<2>(v), 0);
      }

      for (let it = vec.cbegin(); it != vec.cend(); ++it) {
        T v = *it;
        let k = it - vec.cbegin();
        static_assert(std::is_same_v<decltype(k), int64>);
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_EQ(get<1>(v), 0);
        EXPECT_EQ(get<2>(v), 0);
      }

      for (int i = 0; i < 5; ++i) {
        let ptr = vec.data() + i;
        T v = *ptr;
        EXPECT_EQ(get<0>(v), 0);
        EXPECT_EQ(get<1>(v), 0);
        EXPECT_EQ(get<2>(v), 0);

        *ptr = T{i, 2 * i, 3 * i};
        v = *ptr;
        EXPECT_EQ(get<0>(v), i);
        EXPECT_EQ(get<1>(v), 2 * i);
        EXPECT_EQ(get<2>(v), 3 * i);
      }

      {
        TVector vec1(5);
        EXPECT_EQ(vec1.size(), 5);
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(get<0>(v), 0);
          EXPECT_EQ(get<1>(v), 0);
          EXPECT_EQ(get<2>(v), 0);
        }

        vec1.clear();
        EXPECT_EQ(vec1.size(), 0);
      }

      {
        TVector vec1{T{0, 0, 0}, T{1, 2, 3}, T{2, 4, 6}, T{3, 6, 9}, T{4, 8, 12}};
        EXPECT_EQ(vec1.size(), 5);
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(get<0>(v), i);
          EXPECT_EQ(get<1>(v), 2 * i);
          EXPECT_EQ(get<2>(v), 3 * i);
        }

        vec1.clear();
        EXPECT_EQ(vec1.size(), 0);
      }
    });
  }

  // `Vec3<int>`.
  {
    using T = Vec3<int>;
    using TMosaic = Mosaic<T, PatternVec3<int>>;

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
        EXPECT_EQ(v.x(), 0);
        EXPECT_EQ(v.y(), 0);
        EXPECT_EQ(v.z(), 0);

        vec[i] = T{i, 2 * i, 3 * i};
        v = vec[i];
        EXPECT_EQ(v.x(), i);
        EXPECT_EQ(v.y(), 2 * i);
        EXPECT_EQ(v.z(), 3 * i);
      }

      for (let it = vec.begin(); it != vec.end(); ++it) {
        T v = *it;
        let k = it - vec.begin();
        int i = k;
        static_assert(std::is_same_v<decltype(k), int64>);
        EXPECT_EQ(v.x(), i);
        EXPECT_EQ(v.y(), 2 * i);
        EXPECT_EQ(v.z(), 3 * i);

        *it = T{0, 0, 0};
        v = vec[i];
        EXPECT_EQ(v.x(), 0);
        EXPECT_EQ(v.y(), 0);
        EXPECT_EQ(v.z(), 0);
      }

      for (let it = vec.cbegin(); it != vec.cend(); ++it) {
        T v = *it;
        let k = it - vec.cbegin();
        static_assert(std::is_same_v<decltype(k), int64>);
        EXPECT_EQ(v.x(), 0);
        EXPECT_EQ(v.y(), 0);
        EXPECT_EQ(v.z(), 0);
      }

      for (int i = 0; i < 5; ++i) {
        let ptr = vec.data() + i;
        T v = *ptr;
        EXPECT_EQ(v.x(), 0);
        EXPECT_EQ(v.y(), 0);
        EXPECT_EQ(v.z(), 0);

        *ptr = T{i, 2 * i, 3 * i};
        v = *ptr;
        EXPECT_EQ(v.x(), i);
        EXPECT_EQ(v.y(), 2 * i);
        EXPECT_EQ(v.z(), 3 * i);
      }

      {
        TVector vec1(5);
        EXPECT_EQ(vec1.size(), 5);
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(v.x(), 0);
          EXPECT_EQ(v.y(), 0);
          EXPECT_EQ(v.z(), 0);
        }

        vec1.clear();
        EXPECT_EQ(vec1.size(), 0);
      }

      {
        TVector vec1{T{0, 0, 0}, T{1, 2, 3}, T{2, 4, 6}, T{3, 6, 9}, T{4, 8, 12}};
        EXPECT_EQ(vec1.size(), 5);
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(v.x(), i);
          EXPECT_EQ(v.y(), 2 * i);
          EXPECT_EQ(v.z(), 3 * i);
        }

        vec1.clear();
        EXPECT_EQ(vec1.size(), 0);
      }
    });
  }
}

} // namespace ARIA
