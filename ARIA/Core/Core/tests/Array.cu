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

TEST(Vector, Base) {}

} // namespace ARIA
