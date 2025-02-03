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
}

} // namespace ARIA
