#include "ARIA/Let.h"
#include "ARIA/detail/MosaicIterator.h"

#include <gtest/gtest.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

TEST(MosaicIterator, Mosaic) {
  using namespace mosaic::detail;

  // `int, float`.
  {
    using T = Tup<int, float>;
    using TMosaic = Mosaic<T, PatternIF>;

    std::vector<int> is = {0, 1, 2, 3, 4};
    std::array<float, 5> fs = {0.1F, 1.2F, 2.3F, 3.4F, 4.5F};
    const let &isC = is;
    const let &fsC = fs;

    {
      let begin = make_mosaic_iterator<TMosaic>(Tup{is.begin(), fs.begin()});
      let end = make_mosaic_iterator<TMosaic>(Tup{is.end(), fs.end()});
      let beginC = make_mosaic_iterator<TMosaic>(Tup{is.cbegin(), fs.cbegin()});
      let endC = make_mosaic_iterator<TMosaic>(Tup{is.cend(), fs.cend()});
      let data = make_mosaic_pointer<TMosaic>(Tup{is.data(), fs.data()});
      let dataC = make_mosaic_pointer<TMosaic>(Tup{isC.data(), fsC.data()});

      static_assert(Property<decltype(*begin)>);
      static_assert(Property<decltype(*end)>);
      static_assert(Property<decltype(*beginC)>);
      static_assert(Property<decltype(*endC)>);
      static_assert(Property<decltype(*data)>);
      static_assert(Property<decltype(*dataC)>);

      static_assert(is_mosaic_reference_v<decltype(*begin)>);
      static_assert(is_mosaic_reference_v<decltype(*end)>);
      static_assert(is_mosaic_reference_v<decltype(*beginC)>);
      static_assert(is_mosaic_reference_v<decltype(*endC)>);
      static_assert(is_mosaic_reference_v<decltype(*data)>);
      static_assert(is_mosaic_reference_v<decltype(*dataC)>);

      static_assert(MosaicIterator<decltype(begin)>);
      static_assert(MosaicIterator<decltype(end)>);
      static_assert(MosaicIterator<decltype(beginC)>);
      static_assert(MosaicIterator<decltype(endC)>);
      static_assert(MosaicPointer<decltype(data)>);
      static_assert(MosaicPointer<decltype(dataC)>);
      static_assert(!MosaicIterator<decltype(is.begin())>);
      static_assert(!MosaicPointer<decltype(is.data())>);
      static_assert(!MosaicPointer<decltype(isC.data())>);

      static_assert(std::is_same_v<decltype(*begin), decltype(*data)>);
      static_assert(std::is_same_v<decltype(*end), decltype(*data)>);
      static_assert(std::is_same_v<decltype(*beginC), decltype(*dataC)>);
      static_assert(std::is_same_v<decltype(*endC), decltype(*dataC)>);

      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(begin.base().get_iterator_tuple())>,
                                   decltype(Tup{is.begin(), fs.begin()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(end.base().get_iterator_tuple())>,
                                   decltype(Tup{is.end(), fs.end()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(beginC.base().get_iterator_tuple())>,
                                   decltype(Tup{is.cbegin(), fs.cbegin()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(endC.base().get_iterator_tuple())>,
                                   decltype(Tup{is.cend(), fs.cend()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(data.base().get_iterator_tuple())>,
                                   decltype(Tup{is.data(), fs.data()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(dataC.base().get_iterator_tuple())>,
                                   decltype(Tup{isC.data(), fsC.data()})>);

      static_assert(std::is_same_v<mosaic_iterator_2_tup_t<decltype(begin)>, decltype(Tup{is.begin(), fs.begin()})>);
      static_assert(std::is_same_v<mosaic_iterator_2_tup_t<decltype(end)>, decltype(Tup{is.end(), fs.end()})>);
      static_assert(std::is_same_v<mosaic_iterator_2_tup_t<decltype(beginC)>, decltype(Tup{is.cbegin(), fs.cbegin()})>);
      static_assert(std::is_same_v<mosaic_iterator_2_tup_t<decltype(endC)>, decltype(Tup{is.cend(), fs.cend()})>);
      static_assert(std::is_same_v<mosaic_pointer_2_tup_t<decltype(data)>, decltype(Tup{is.data(), fs.data()})>);
      static_assert(std::is_same_v<mosaic_pointer_2_tup_t<decltype(dataC)>, decltype(Tup{isC.data(), fsC.data()})>);

      EXPECT_EQ(make_mosaic_iterator<TMosaic>(MosaicIterator2Tup(begin)), begin);
      EXPECT_EQ(make_mosaic_iterator<TMosaic>(MosaicIterator2Tup(end)), end);
      EXPECT_EQ(make_mosaic_iterator<TMosaic>(MosaicIterator2Tup(beginC)), beginC);
      EXPECT_EQ(make_mosaic_iterator<TMosaic>(MosaicIterator2Tup(endC)), endC);
      EXPECT_EQ(make_mosaic_pointer<TMosaic>(MosaicPointer2Tup(data)), data);
      EXPECT_EQ(make_mosaic_pointer<TMosaic>(MosaicPointer2Tup(dataC)), dataC);
    }

    {
      let begin = make_mosaic_iterator<TMosaic>(Tup{is.begin(), fs.begin()});
      let end = make_mosaic_iterator<TMosaic>(Tup{is.end(), fs.end()});
      let data = make_mosaic_pointer<TMosaic>(Tup{is.data(), fs.data()});

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

      EXPECT_EQ(end - begin, 5);
      for (size_t i = 0; i < end - begin; ++i) {
        let ptr = data + i;
        let v = Let(*ptr);
        static_assert(std::is_same_v<decltype(v), T>);

        if (ptr == data + 0) {
          EXPECT_EQ(get<0>(v), 10);
          EXPECT_FLOAT_EQ(get<1>(v), 10.11F);
        } else if (ptr == data + 1) {
          EXPECT_EQ(get<0>(v), 11);
          EXPECT_FLOAT_EQ(get<1>(v), 11.21F);
        } else if (ptr == data + 2) {
          EXPECT_EQ(get<0>(v), 12);
          EXPECT_FLOAT_EQ(get<1>(v), 12.31F);
        } else if (ptr == data + 3) {
          EXPECT_EQ(get<0>(v), 13);
          EXPECT_FLOAT_EQ(get<1>(v), 13.41F);
        } else if (ptr == data + 4) {
          EXPECT_EQ(get<0>(v), 14);
          EXPECT_FLOAT_EQ(get<1>(v), 14.51F);
        }
      }
    }

    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(is[i], i + 10);
      EXPECT_TRUE(std::abs(fs[i] - (i + (i + 1) * 0.1 + 10.01)) < 1e-6);
    }
  }

  // `int, int, int`.
  {
    using T = Tec<int, int, int>;
    using TMosaic = Mosaic<T, PatternIII>;

    std::array<int, 5> is0 = {0, 1, 2, 3, 4};
    thrust::host_vector<int> is1 = {0, -2, -4, -6, -8};
    thrust::device_vector<int> is2 = {0, 3, 6, 9, 12};
    const let &is0C = is0;
    const let &is1C = is1;
    const let &is2C = is2;

    {
      let begin = make_mosaic_iterator<TMosaic>(Tup{is0.begin(), is1.begin(), is2.begin()});
      let end = make_mosaic_iterator<TMosaic>(Tup{is0.end(), is1.end(), is2.end()});
      let beginC = make_mosaic_iterator<TMosaic>(Tup{is0.cbegin(), is1.cbegin(), is2.cbegin()});
      let endC = make_mosaic_iterator<TMosaic>(Tup{is0.cend(), is1.cend(), is2.cend()});
      let data = make_mosaic_pointer<TMosaic>(Tup{is0.data(), is1.data(), is2.data()});
      let dataC = make_mosaic_pointer<TMosaic>(Tup{is0C.data(), is1C.data(), is2C.data()});

      static_assert(Property<decltype(*begin)>);
      static_assert(Property<decltype(*end)>);
      static_assert(Property<decltype(*beginC)>);
      static_assert(Property<decltype(*endC)>);
      static_assert(Property<decltype(*data)>);
      static_assert(Property<decltype(*dataC)>);

      static_assert(is_mosaic_reference_v<decltype(*begin)>);
      static_assert(is_mosaic_reference_v<decltype(*end)>);
      static_assert(is_mosaic_reference_v<decltype(*beginC)>);
      static_assert(is_mosaic_reference_v<decltype(*endC)>);
      static_assert(is_mosaic_reference_v<decltype(*data)>);
      static_assert(is_mosaic_reference_v<decltype(*dataC)>);

      static_assert(MosaicIterator<decltype(begin)>);
      static_assert(MosaicIterator<decltype(end)>);
      static_assert(MosaicIterator<decltype(beginC)>);
      static_assert(MosaicIterator<decltype(endC)>);
      static_assert(MosaicPointer<decltype(data)>);
      static_assert(MosaicPointer<decltype(dataC)>);
      static_assert(!MosaicIterator<decltype(is0.begin())>);
      static_assert(!MosaicPointer<decltype(is0.data())>);
      static_assert(!MosaicPointer<decltype(is0C.data())>);

      static_assert(std::is_same_v<decltype(*begin), decltype(*data)>);
      static_assert(std::is_same_v<decltype(*end), decltype(*data)>);
      static_assert(std::is_same_v<decltype(*beginC), decltype(*dataC)>);
      static_assert(std::is_same_v<decltype(*endC), decltype(*dataC)>);

      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(begin.base().get_iterator_tuple())>,
                                   decltype(Tup{is0.begin(), is1.begin(), is2.begin()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(end.base().get_iterator_tuple())>,
                                   decltype(Tup{is0.end(), is1.end(), is2.end()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(beginC.base().get_iterator_tuple())>,
                                   decltype(Tup{is0.cbegin(), is1.cbegin(), is2.cbegin()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(endC.base().get_iterator_tuple())>,
                                   decltype(Tup{is0.cend(), is1.cend(), is2.cend()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(data.base().get_iterator_tuple())>,
                                   decltype(Tup{is0.data(), is1.data(), is2.data()})>);
      static_assert(std::is_same_v<thrust_tuple_2_tup_t<decltype(dataC.base().get_iterator_tuple())>,
                                   decltype(Tup{is0C.data(), is1C.data(), is2C.data()})>);

      static_assert(std::is_same_v<mosaic_iterator_2_tup_t<decltype(begin)>, //
                                   decltype(Tup{is0.begin(), is1.begin(), is2.begin()})>);
      static_assert(std::is_same_v<mosaic_iterator_2_tup_t<decltype(end)>, //
                                   decltype(Tup{is0.end(), is1.end(), is2.end()})>);
      static_assert(std::is_same_v<mosaic_iterator_2_tup_t<decltype(beginC)>, //
                                   decltype(Tup{is0.cbegin(), is1.cbegin(), is2.cbegin()})>);
      static_assert(std::is_same_v<mosaic_iterator_2_tup_t<decltype(endC)>, //
                                   decltype(Tup{is0.cend(), is1.cend(), is2.cend()})>);
      static_assert(std::is_same_v<mosaic_pointer_2_tup_t<decltype(data)>, //
                                   decltype(Tup{is0.data(), is1.data(), is2.data()})>);
      static_assert(std::is_same_v<mosaic_pointer_2_tup_t<decltype(dataC)>, //
                                   decltype(Tup{is0C.data(), is1C.data(), is2C.data()})>);

      EXPECT_EQ(make_mosaic_iterator<TMosaic>(MosaicIterator2Tup(begin)), begin);
      EXPECT_EQ(make_mosaic_iterator<TMosaic>(MosaicIterator2Tup(end)), end);
      EXPECT_EQ(make_mosaic_iterator<TMosaic>(MosaicIterator2Tup(beginC)), beginC);
      EXPECT_EQ(make_mosaic_iterator<TMosaic>(MosaicIterator2Tup(endC)), endC);
      EXPECT_EQ(make_mosaic_pointer<TMosaic>(MosaicPointer2Tup(data)), data);
      EXPECT_EQ(make_mosaic_pointer<TMosaic>(MosaicPointer2Tup(dataC)), dataC);
    }

    {
      let begin = make_mosaic_iterator<TMosaic>(Tup{is0.begin(), is1.begin(), is2.begin()});
      let end = make_mosaic_iterator<TMosaic>(Tup{is0.end(), is1.end(), is2.end()});
      let data = make_mosaic_pointer<TMosaic>(Tup{is0.data(), is1.data(), is2.data()});

      for (let it = begin; it != end; ++it) {
        let v = Let(*it);
        static_assert(std::is_same_v<decltype(v), T>);

        if (it == begin + 0) {
          EXPECT_EQ(get<0>(v), 0);
          EXPECT_EQ(get<1>(v), 0);
          EXPECT_EQ(get<2>(v), 0);
        } else if (it == begin + 1) {
          EXPECT_EQ(get<0>(v), 1);
          EXPECT_EQ(get<1>(v), -2);
          EXPECT_EQ(get<2>(v), 3);
        } else if (it == begin + 2) {
          EXPECT_EQ(get<0>(v), 2);
          EXPECT_EQ(get<1>(v), -4);
          EXPECT_EQ(get<2>(v), 6);
        } else if (it == begin + 3) {
          EXPECT_EQ(get<0>(v), 3);
          EXPECT_EQ(get<1>(v), -6);
          EXPECT_EQ(get<2>(v), 9);
        } else if (it == begin + 4) {
          EXPECT_EQ(get<0>(v), 4);
          EXPECT_EQ(get<1>(v), -8);
          EXPECT_EQ(get<2>(v), 12);
        }

        *it = {1, 10, 100};
        *it *= 5;
        v = *it;

        EXPECT_EQ(get<0>(v), 5);
        EXPECT_EQ(get<1>(v), 50);
        EXPECT_EQ(get<2>(v), 500);
      }

      EXPECT_EQ(end - begin, 5);
      for (size_t i = 0; i < end - begin; ++i) {
        let ptr = data + i;
        let v = Let(*ptr);
        static_assert(std::is_same_v<decltype(v), T>);

        EXPECT_EQ(get<0>(v), 5);
        EXPECT_EQ(get<1>(v), 50);
        EXPECT_EQ(get<2>(v), 500);
      }
    }

    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(is0[i], 5);
      EXPECT_EQ(is1[i], 50);
      EXPECT_EQ(is2[i], 500);
    }
  }
}

TEST(MosaicIterator, NonMosaic) {
  using namespace mosaic::detail;

  // `int`.
  std::array<int, 5> is = {0, 1, 2, 3, 4};
  const let &isC = is;

  {
    let begin = make_mosaic_iterator<int>(Tup{is.begin()});
    let end = make_mosaic_iterator<int>(Tup{is.end()});
    let beginC = make_mosaic_iterator<int>(Tup{is.cbegin()});
    let endC = make_mosaic_iterator<int>(Tup{is.cend()});
    let data = make_mosaic_pointer<int>(Tup{is.data()});
    let dataC = make_mosaic_pointer<int>(Tup{isC.data()});

    static_assert(!Property<decltype(*begin)>);
    static_assert(!Property<decltype(*end)>);
    static_assert(!Property<decltype(*beginC)>);
    static_assert(!Property<decltype(*endC)>);
    static_assert(!Property<decltype(*data)>);
    static_assert(!Property<decltype(*dataC)>);

    static_assert(!MosaicIterator<decltype(begin)>);
    static_assert(!MosaicIterator<decltype(end)>);
    static_assert(!MosaicIterator<decltype(beginC)>);
    static_assert(!MosaicIterator<decltype(endC)>);
    static_assert(!MosaicPointer<decltype(data)>);
    static_assert(!MosaicPointer<decltype(dataC)>);

    static_assert(std::is_same_v<decltype(begin), decltype(is.begin())>);
    static_assert(std::is_same_v<decltype(end), decltype(is.end())>);
    static_assert(std::is_same_v<decltype(beginC), decltype(is.cbegin())>);
    static_assert(std::is_same_v<decltype(endC), decltype(is.cend())>);
    static_assert(std::is_same_v<decltype(data), decltype(is.data())>);
    static_assert(std::is_same_v<decltype(dataC), decltype(isC.data())>);

    static_assert(std::is_same_v<decltype(*begin), int &>);
    static_assert(std::is_same_v<decltype(*end), int &>);
    static_assert(std::is_same_v<decltype(*beginC), const int &>);
    static_assert(std::is_same_v<decltype(*endC), const int &>);
    static_assert(std::is_same_v<decltype(*data), int &>);
    static_assert(std::is_same_v<decltype(*dataC), const int &>);

    static_assert(std::is_same_v<decltype(*begin), decltype(*data)>);
    static_assert(std::is_same_v<decltype(*end), decltype(*data)>);
    static_assert(std::is_same_v<decltype(*beginC), decltype(*dataC)>);
    static_assert(std::is_same_v<decltype(*endC), decltype(*dataC)>);
  }

  {
    let begin = make_mosaic_iterator<int>(Tup{is.begin()});
    let end = make_mosaic_iterator<int>(Tup{is.end()});
    let data = make_mosaic_pointer<int>(Tup{is.data()});

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

    EXPECT_EQ(end - begin, 5);
    for (size_t i = 0; i < end - begin; ++i) {
      let ptr = data + i;
      let v = *ptr;
      static_assert(std::is_same_v<decltype(v), int>);

      if (ptr == data + 0) {
        EXPECT_EQ(v, 10);
      } else if (ptr == data + 1) {
        EXPECT_EQ(v, 11);
      } else if (ptr == data + 2) {
        EXPECT_EQ(v, 12);
      } else if (ptr == data + 3) {
        EXPECT_EQ(v, 13);
      } else if (ptr == data + 4) {
        EXPECT_EQ(v, 14);
      }
    }
  }

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(is[i], i + 10);
  }
}

TEST(MosaicIterator, Copy) {
  using namespace mosaic::detail;

  // `int, float`.
  {
    using T = Tup<int, float>;
    using TMosaic = Mosaic<T, PatternIF>;

    const std::vector<int> srcIs = {0, 1, 2, 3, 4};
    const std::array<float, 5> srcFs = {0.1F, 1.2F, 2.3F, 3.4F, 4.5F};
    std::array<int, 5> dstIs;
    std::vector<float> dstFs(5);

    let srcBegin = make_mosaic_iterator<TMosaic>(Tup{srcIs.cbegin(), srcFs.cbegin()});
    let srcEnd = make_mosaic_iterator<TMosaic>(Tup{srcIs.cend(), srcFs.cend()});
    let dstBegin = make_mosaic_iterator<TMosaic>(Tup{dstIs.begin(), dstFs.begin()});
    let dstEnd = make_mosaic_iterator<TMosaic>(Tup{dstIs.end(), dstFs.end()});

    let res = copy(srcBegin, srcEnd, dstBegin);
    EXPECT_EQ(res, dstEnd);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(dstIs[i], i);
      EXPECT_FLOAT_EQ(dstFs[i], i + (i + 1) * 0.1F);
    }
  }

  // `int, int, int`.
  {
    using T = Tec<int, int, int>;
    using TMosaic = Mosaic<T, PatternIII>;

    const std::array<int, 5> srcIs0 = {0, 1, 2, 3, 4};
    const thrust::host_vector<int> srcIs1 = {0, -2, -4, -6, -8};
    const thrust::device_vector<int> srcIs2 = {0, 3, 6, 9, 12};
    thrust::device_vector<int> dstIs0(5);
    thrust::device_vector<int> dstIs1(5);
    thrust::host_vector<int> dstIs2(5);

    let srcBegin = make_mosaic_iterator<TMosaic>(Tup{srcIs0.cbegin(), srcIs1.cbegin(), srcIs2.cbegin()});
    let srcEnd = make_mosaic_iterator<TMosaic>(Tup{srcIs0.cend(), srcIs1.cend(), srcIs2.cend()});
    let dstBegin = make_mosaic_iterator<TMosaic>(Tup{dstIs0.begin(), dstIs1.begin(), dstIs2.begin()});
    let dstEnd = make_mosaic_iterator<TMosaic>(Tup{dstIs0.end(), dstIs1.end(), dstIs2.end()});

    let res = copy(srcBegin, srcEnd, dstBegin);
    EXPECT_EQ(res, dstEnd);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(dstIs0[i], i);
      EXPECT_EQ(dstIs1[i], -2 * i);
      EXPECT_EQ(dstIs2[i], 3 * i);
    }
  }

  // `int`.
  {
    const thrust::device_vector<int> srcIs = {0, 1, 2, 3, 4};
    std::array<int, 5> dstIs;

    let srcBegin = make_mosaic_iterator<int>(Tup{srcIs.begin()});
    let srcEnd = make_mosaic_iterator<int>(Tup{srcIs.end()});
    let dstBegin = make_mosaic_iterator<int>(Tup{dstIs.begin()});
    let dstEnd = make_mosaic_iterator<int>(Tup{dstIs.end()});

    let res = copy(srcBegin, srcEnd, dstBegin);
    EXPECT_EQ(res, dstEnd);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(dstIs[i], i);
    }
  }
}

} // namespace ARIA
