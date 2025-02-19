#include "ARIA/Array.h"
#include "ARIA/Let.h"

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

template <typename T, uint size>
class Vec {
public:
  Vec() = default;

  ARIA_HOST_DEVICE Vec(std::initializer_list<T> list) {
    ARIA_ASSERT(list.size() == size);
    uint i = 0;
    for (const T &v : list) {
      v_[i] = v;
      ++i;
    }
  }

  ARIA_COPY_MOVE_ABILITY(Vec, default, default);

  ARIA_HOST_DEVICE const T &operator[](uint i) const { return v_[i]; }

  ARIA_HOST_DEVICE T &operator[](uint i) { return v_[i]; }

private:
  T v_[size];
};

template <typename T, uint size>
struct PatternVec {
  T v[size];
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

template <typename T, uint size>
struct Mosaic<Vec<T, size>, PatternVec<T, size>> {
  PatternVec<T, size> operator()(const Vec<T, size> &v) const {
    PatternVec<T, size> res;
    ForEach<size>([&]<auto i>() { res.v[i] = v[i]; });
    return res;
  }

  Vec<T, size> operator()(const PatternVec<T, size> &v) const {
    Vec<T, size> res;
    ForEach<size>([&]<auto i>() { res[i] = v.v[i]; });
    return res;
  }
};

TEST(Array, Base) {
  // Mosaic.
  {
    ForEach<MakeTypeArray<                          //
        Mosaic<Tup<int, float>, PatternIF>,         //
        Mosaic<Tup<int, int, int>, PatternIII>,     //
        Mosaic<Vec3<int>, PatternVec3<int>>,        //
        Mosaic<Vec3<float>, PatternVec3<float>>,    //
        Mosaic<Vec<int, 3>, PatternVec<int, 3>>,    //
        Mosaic<Vec<float, 4>, PatternVec<float, 4>> //
        >>([]<typename TMosaic>() {
      static_assert(std::is_same_v<Array<TMosaic, 5>, mosaic::detail::MosaicArray<TMosaic, 5>>);
    });
  }

  // Non-mosaic.
  {
    ForEach<MakeTypeArray<                          //
        int,                                        //
        Tup<int, float, double>,                    //
        Tup<Tec2i<int, Int<5>>, Tec<float, double>> //
        >>([]<typename T>() { static_assert(std::is_same_v<Array<T, 5>, cuda::std::array<T, 5>>); });
  }
}

TEST(Array, Methods) {
  // `int, float`.
  {
    using T = Tup<int, float>;
    using TMosaic = Mosaic<T, PatternIF>;

    ForEach<MakeTypeArray< //
        Array<TMosaic, 5>, //
        Array<T, 5>        //
        >>([]<typename TArray>() {
      TArray vec{};
      static_assert(vec.size() == 5);

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
        TArray vec1{{T{0, 0.1F}, T{1, 1.2F}, T{2, 2.3F}, T{3, 3.4F}, T{4, 4.5F}}};
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(get<0>(v), i);
          EXPECT_FLOAT_EQ(get<1>(v), i + (i + 1) * 0.1F);
        }
      }
    });
  }

  // `int, int, int`.
  {
    using T = Tec<int, int, int>;
    using TMosaic = Mosaic<T, PatternIII>;

    ForEach<MakeTypeArray< //
        Array<TMosaic, 5>, //
        Array<T, 5>        //
        >>([]<typename TArray>() {
      TArray vec{};
      static_assert(vec.size() == 5);

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
        TArray vec1{{T{0, 0, 0}, T{1, 2, 3}, T{2, 4, 6}, T{3, 6, 9}, T{4, 8, 12}}};
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(get<0>(v), i);
          EXPECT_EQ(get<1>(v), 2 * i);
          EXPECT_EQ(get<2>(v), 3 * i);
        }
      }
    });
  }

  // `Vec3<int>`.
  {
    using T = Vec3<int>;
    using TMosaic = Mosaic<T, PatternVec3<int>>;

    ForEach<MakeTypeArray< //
        Array<TMosaic, 5>, //
        Array<T, 5>        //
        >>([]<typename TArray>() {
      TArray vec{};
      static_assert(vec.size() == 5);

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
        TArray vec1{{T{0, 0, 0}, T{1, 2, 3}, T{2, 4, 6}, T{3, 6, 9}, T{4, 8, 12}}};
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(v.x(), i);
          EXPECT_EQ(v.y(), 2 * i);
          EXPECT_EQ(v.z(), 3 * i);
        }
      }
    });
  }

  // `Vec<int, 3>`.
  {
    using T = Vec<int, 3>;
    using TMosaic = Mosaic<T, PatternVec<int, 3>>;

    ForEach<MakeTypeArray< //
        Array<TMosaic, 5>, //
        Array<T, 5>        //
        >>([]<typename TArray>() {
      TArray vec{};
      static_assert(vec.size() == 5);

      for (int i = 0; i < 5; ++i) {
        T v = vec[i];
        EXPECT_EQ(v[0], 0);
        EXPECT_EQ(v[1], 0);
        EXPECT_EQ(v[2], 0);

        vec[i] = T{i, 2 * i, 3 * i};
        v = vec[i];
        EXPECT_EQ(v[0], i);
        EXPECT_EQ(v[1], 2 * i);
        EXPECT_EQ(v[2], 3 * i);
      }

      for (let it = vec.begin(); it != vec.end(); ++it) {
        T v = *it;
        let k = it - vec.begin();
        int i = k;
        static_assert(std::is_same_v<decltype(k), int64>);
        EXPECT_EQ(v[0], i);
        EXPECT_EQ(v[1], 2 * i);
        EXPECT_EQ(v[2], 3 * i);

        *it = T{0, 0, 0};
        v = vec[i];
        EXPECT_EQ(v[0], 0);
        EXPECT_EQ(v[1], 0);
        EXPECT_EQ(v[2], 0);
      }

      for (let it = vec.cbegin(); it != vec.cend(); ++it) {
        T v = *it;
        let k = it - vec.cbegin();
        static_assert(std::is_same_v<decltype(k), int64>);
        EXPECT_EQ(v[0], 0);
        EXPECT_EQ(v[1], 0);
        EXPECT_EQ(v[2], 0);
      }

      for (int i = 0; i < 5; ++i) {
        let ptr = vec.data() + i;
        T v = *ptr;
        EXPECT_EQ(v[0], 0);
        EXPECT_EQ(v[1], 0);
        EXPECT_EQ(v[2], 0);

        *ptr = T{i, 2 * i, 3 * i};
        v = *ptr;
        EXPECT_EQ(v[0], i);
        EXPECT_EQ(v[1], 2 * i);
        EXPECT_EQ(v[2], 3 * i);
      }

      {
        TArray vec1{{T{0, 0, 0}, T{1, 2, 3}, T{2, 4, 6}, T{3, 6, 9}, T{4, 8, 12}}};
        EXPECT_EQ(vec1.size(), 5);
        for (int i = 0; i < 5; ++i) {
          T v = vec1[i];
          EXPECT_EQ(v[0], i);
          EXPECT_EQ(v[1], 2 * i);
          EXPECT_EQ(v[2], 3 * i);
        }
      }
    });
  }
}

} // namespace ARIA
