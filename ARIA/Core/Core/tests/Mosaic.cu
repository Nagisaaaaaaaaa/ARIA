#include "ARIA/Launcher.h"
#include "ARIA/Mosaic.h"
#include "ARIA/Property.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct Test0Member {};

// TODO: Currently not supported by `boost::pfr`.
struct Test1Member {
  int v = 5;
};

struct Test1ArrayMember {
  int v[3] = {5, 6, 7};
};

struct Test2Members {
  int v0 = 5;
  double v1 = 6;
};

struct TestPrivateMembers {
private:
  int v_ = 5;
};

struct TestInheritanceBase {
  int v0 = 5;
};

struct TestInheritance : public TestInheritanceBase {
  double v1 = 6;
};

struct TestRecursion0Member {
  int v0 = 5;

  struct {
  } s0;
};

// TODO: Currently not supported by `boost::pfr`.
struct TestRecursion1Member {
  int v0 = 5;

  struct {
    double v1 = 6;
  } s0;
};

struct TestRecursion2Members {
  int v0 = 5;

  struct {
    double v1 = 6;
    int *v2 = nullptr;
  } s0;
};

struct TestRecursionComplex {
  int v0 = 5, v1 = 6, v2 = 7;

  struct {
    double vs0 = 8;
    int *vs1 = nullptr;

    struct {
      int64 vss0 = 9;
      float vss1 = 10;
    } ss0, ss1;

    double *vs2 = nullptr;
  } s0, s1;

  double v3 = 11, v4 = 12;

  struct {
    float vs0 = 13;
    int *vs1 = nullptr;
    double *vs2 = nullptr;
    float *vs3 = nullptr;
  } s2, s3, s4;

  int *v5 = nullptr;
  double *v6 = nullptr;
  float *v7 = nullptr;
};

void TestGetRecursive() {
  using namespace mosaic::detail;

  {
    int v = 5;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(int{})), int>);
    EXPECT_EQ(get_recursive<0>(v), 5);
    EXPECT_EQ(get_recursive<0>(int{5}), 5);
    static_assert(get_recursive<0>(int{5}) == 5);
  }

  Launcher(1, [] ARIA_DEVICE(int i) {
    int v = 5;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(int{})), int>);
    ARIA_ASSERT(get_recursive<0>(v) == 5);
    ARIA_ASSERT(get_recursive<0>(int{5}) == 5);
    static_assert(get_recursive<0>(int{5}) == 5);
  }).Launch();
  cuda::device::current::get().synchronize();

  {
    int v[3] = {5, 6, 7};
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(std::declval<int[3]>())), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(std::declval<int[3]>())), int>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(std::declval<int[3]>())), int>);
    EXPECT_EQ(get_recursive<0>(v), 5);
    EXPECT_EQ(get_recursive<1>(v), 6);
    EXPECT_EQ(get_recursive<2>(v), 7);
  }

  Launcher(1, [] ARIA_DEVICE(int i) {
    int v[3] = {5, 6, 7};
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(std::declval<int[3]>())), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(std::declval<int[3]>())), int>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(std::declval<int[3]>())), int>);
    ARIA_ASSERT(get_recursive<0>(v) == 5);
    ARIA_ASSERT(get_recursive<1>(v) == 6);
    ARIA_ASSERT(get_recursive<2>(v) == 7);
  }).Launch();
  cuda::device::current::get().synchronize();

  {
    Test1ArrayMember v;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(Test1ArrayMember{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(Test1ArrayMember{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(Test1ArrayMember{})), int>);
    EXPECT_EQ(get_recursive<0>(v), 5);
    EXPECT_EQ(get_recursive<1>(v), 6);
    EXPECT_EQ(get_recursive<2>(v), 7);
    EXPECT_EQ(get_recursive<0>(Test1ArrayMember{}), 5);
    EXPECT_EQ(get_recursive<1>(Test1ArrayMember{}), 6);
    EXPECT_EQ(get_recursive<2>(Test1ArrayMember{}), 7);
    static_assert(get_recursive<0>(Test1ArrayMember{}) == 5);
    static_assert(get_recursive<1>(Test1ArrayMember{}) == 6);
    static_assert(get_recursive<2>(Test1ArrayMember{}) == 7);
  }

  Launcher(1, [] ARIA_DEVICE(int i) {
    Test1ArrayMember v;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(Test1ArrayMember{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(Test1ArrayMember{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(Test1ArrayMember{})), int>);
    ARIA_ASSERT(get_recursive<0>(v) == 5);
    ARIA_ASSERT(get_recursive<1>(v) == 6);
    ARIA_ASSERT(get_recursive<2>(v) == 7);
    ARIA_ASSERT(get_recursive<0>(Test1ArrayMember{}) == 5);
    ARIA_ASSERT(get_recursive<1>(Test1ArrayMember{}) == 6);
    ARIA_ASSERT(get_recursive<2>(Test1ArrayMember{}) == 7);
    static_assert(get_recursive<0>(Test1ArrayMember{}) == 5);
    static_assert(get_recursive<1>(Test1ArrayMember{}) == 6);
    static_assert(get_recursive<2>(Test1ArrayMember{}) == 7);
  }).Launch();
  cuda::device::current::get().synchronize();

  {
    Test2Members v;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(Test2Members{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(Test2Members{})), double>);
    EXPECT_EQ(get_recursive<0>(v), 5);
    EXPECT_EQ(get_recursive<1>(v), 6);
    EXPECT_EQ(get_recursive<0>(Test2Members{}), 5);
    EXPECT_EQ(get_recursive<1>(Test2Members{}), 6);
    static_assert(get_recursive<0>(Test2Members{}) == 5);
    static_assert(get_recursive<1>(Test2Members{}) == 6);
  }

  Launcher(1, [] ARIA_DEVICE(int i) {
    Test2Members v;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(Test2Members{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(Test2Members{})), double>);
    ARIA_ASSERT(get_recursive<0>(v) == 5);
    ARIA_ASSERT(get_recursive<1>(v) == 6);
    ARIA_ASSERT(get_recursive<0>(Test2Members{}) == 5);
    ARIA_ASSERT(get_recursive<1>(Test2Members{}) == 6);
    static_assert(get_recursive<0>(Test2Members{}) == 5);
    static_assert(get_recursive<1>(Test2Members{}) == 6);
  }).Launch();
  cuda::device::current::get().synchronize();

  {
    TestRecursion2Members v;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(TestRecursion2Members{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(TestRecursion2Members{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(TestRecursion2Members{})), int *>);
    EXPECT_EQ(get_recursive<0>(v), 5);
    EXPECT_EQ(get_recursive<1>(v), 6);
    EXPECT_EQ(get_recursive<2>(v), nullptr);
    EXPECT_EQ(get_recursive<0>(TestRecursion2Members{}), 5);
    EXPECT_EQ(get_recursive<1>(TestRecursion2Members{}), 6);
    EXPECT_EQ(get_recursive<2>(TestRecursion2Members{}), nullptr);
    static_assert(get_recursive<0>(TestRecursion2Members{}) == 5);
    static_assert(get_recursive<1>(TestRecursion2Members{}) == 6);
    static_assert(get_recursive<2>(TestRecursion2Members{}) == nullptr);
  }

  Launcher(1, [] ARIA_DEVICE(int i) {
    TestRecursion2Members v;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(TestRecursion2Members{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(TestRecursion2Members{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(TestRecursion2Members{})), int *>);
    ARIA_ASSERT(get_recursive<0>(v) == 5);
    ARIA_ASSERT(get_recursive<1>(v) == 6);
    ARIA_ASSERT(get_recursive<2>(v) == nullptr);
    ARIA_ASSERT(get_recursive<0>(TestRecursion2Members{}) == 5);
    ARIA_ASSERT(get_recursive<1>(TestRecursion2Members{}) == 6);
    ARIA_ASSERT(get_recursive<2>(TestRecursion2Members{}) == nullptr);
    static_assert(get_recursive<0>(TestRecursion2Members{}) == 5);
    static_assert(get_recursive<1>(TestRecursion2Members{}) == 6);
    static_assert(get_recursive<2>(TestRecursion2Members{}) == nullptr);
  }).Launch();
  cuda::device::current::get().synchronize();

  {
    TestRecursionComplex v;

    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<3>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<4>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<5>(v)), int64 &>);
    static_assert(std::is_same_v<decltype(get_recursive<6>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<7>(v)), int64 &>);
    static_assert(std::is_same_v<decltype(get_recursive<8>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<9>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<10>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<11>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<12>(v)), int64 &>);
    static_assert(std::is_same_v<decltype(get_recursive<13>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<14>(v)), int64 &>);
    static_assert(std::is_same_v<decltype(get_recursive<15>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<16>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<17>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<18>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<19>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<20>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<21>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<22>(v)), float *&>);
    static_assert(std::is_same_v<decltype(get_recursive<23>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<24>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<25>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<26>(v)), float *&>);
    static_assert(std::is_same_v<decltype(get_recursive<27>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<28>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<29>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<30>(v)), float *&>);
    static_assert(std::is_same_v<decltype(get_recursive<31>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<32>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<33>(v)), float *&>);

    static_assert(std::is_same_v<decltype(get_recursive<0>(TestRecursionComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(TestRecursionComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(TestRecursionComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<3>(TestRecursionComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<4>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<5>(TestRecursionComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<6>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<7>(TestRecursionComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<8>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<9>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<10>(TestRecursionComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<11>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<12>(TestRecursionComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<13>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<14>(TestRecursionComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<15>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<16>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<17>(TestRecursionComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<18>(TestRecursionComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<19>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<20>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<21>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<22>(TestRecursionComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<23>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<24>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<25>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<26>(TestRecursionComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<27>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<28>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<29>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<30>(TestRecursionComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<31>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<32>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<33>(TestRecursionComplex{})), float *>);

    EXPECT_EQ(get_recursive<0>(v), 5);
    EXPECT_EQ(get_recursive<1>(v), 6);
    EXPECT_EQ(get_recursive<2>(v), 7);
    EXPECT_EQ(get_recursive<3>(v), 8);
    EXPECT_EQ(get_recursive<4>(v), nullptr);
    EXPECT_EQ(get_recursive<5>(v), 9);
    EXPECT_EQ(get_recursive<6>(v), 10);
    EXPECT_EQ(get_recursive<7>(v), 9);
    EXPECT_EQ(get_recursive<8>(v), 10);
    EXPECT_EQ(get_recursive<9>(v), nullptr);
    EXPECT_EQ(get_recursive<10>(v), 8);
    EXPECT_EQ(get_recursive<11>(v), nullptr);
    EXPECT_EQ(get_recursive<12>(v), 9);
    EXPECT_EQ(get_recursive<13>(v), 10);
    EXPECT_EQ(get_recursive<14>(v), 9);
    EXPECT_EQ(get_recursive<15>(v), 10);
    EXPECT_EQ(get_recursive<16>(v), nullptr);
    EXPECT_EQ(get_recursive<17>(v), 11);
    EXPECT_EQ(get_recursive<18>(v), 12);
    EXPECT_EQ(get_recursive<19>(v), 13);
    EXPECT_EQ(get_recursive<20>(v), nullptr);
    EXPECT_EQ(get_recursive<21>(v), nullptr);
    EXPECT_EQ(get_recursive<22>(v), nullptr);
    EXPECT_EQ(get_recursive<23>(v), 13);
    EXPECT_EQ(get_recursive<24>(v), nullptr);
    EXPECT_EQ(get_recursive<25>(v), nullptr);
    EXPECT_EQ(get_recursive<26>(v), nullptr);
    EXPECT_EQ(get_recursive<27>(v), 13);
    EXPECT_EQ(get_recursive<28>(v), nullptr);
    EXPECT_EQ(get_recursive<29>(v), nullptr);
    EXPECT_EQ(get_recursive<30>(v), nullptr);
    EXPECT_EQ(get_recursive<31>(v), nullptr);
    EXPECT_EQ(get_recursive<32>(v), nullptr);
    EXPECT_EQ(get_recursive<33>(v), nullptr);

    EXPECT_EQ(get_recursive<0>(TestRecursionComplex{}), 5);
    EXPECT_EQ(get_recursive<1>(TestRecursionComplex{}), 6);
    EXPECT_EQ(get_recursive<2>(TestRecursionComplex{}), 7);
    EXPECT_EQ(get_recursive<3>(TestRecursionComplex{}), 8);
    EXPECT_EQ(get_recursive<4>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<5>(TestRecursionComplex{}), 9);
    EXPECT_EQ(get_recursive<6>(TestRecursionComplex{}), 10);
    EXPECT_EQ(get_recursive<7>(TestRecursionComplex{}), 9);
    EXPECT_EQ(get_recursive<8>(TestRecursionComplex{}), 10);
    EXPECT_EQ(get_recursive<9>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<10>(TestRecursionComplex{}), 8);
    EXPECT_EQ(get_recursive<11>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<12>(TestRecursionComplex{}), 9);
    EXPECT_EQ(get_recursive<13>(TestRecursionComplex{}), 10);
    EXPECT_EQ(get_recursive<14>(TestRecursionComplex{}), 9);
    EXPECT_EQ(get_recursive<15>(TestRecursionComplex{}), 10);
    EXPECT_EQ(get_recursive<16>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<17>(TestRecursionComplex{}), 11);
    EXPECT_EQ(get_recursive<18>(TestRecursionComplex{}), 12);
    EXPECT_EQ(get_recursive<19>(TestRecursionComplex{}), 13);
    EXPECT_EQ(get_recursive<20>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<21>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<22>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<23>(TestRecursionComplex{}), 13);
    EXPECT_EQ(get_recursive<24>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<25>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<26>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<27>(TestRecursionComplex{}), 13);
    EXPECT_EQ(get_recursive<28>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<29>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<30>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<31>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<32>(TestRecursionComplex{}), nullptr);
    EXPECT_EQ(get_recursive<33>(TestRecursionComplex{}), nullptr);

    static_assert(get_recursive<0>(TestRecursionComplex{}) == 5);
    static_assert(get_recursive<1>(TestRecursionComplex{}) == 6);
    static_assert(get_recursive<2>(TestRecursionComplex{}) == 7);
    static_assert(get_recursive<3>(TestRecursionComplex{}) == 8);
    static_assert(get_recursive<4>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<5>(TestRecursionComplex{}) == 9);
    static_assert(get_recursive<6>(TestRecursionComplex{}) == 10);
    static_assert(get_recursive<7>(TestRecursionComplex{}) == 9);
    static_assert(get_recursive<8>(TestRecursionComplex{}) == 10);
    static_assert(get_recursive<9>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<10>(TestRecursionComplex{}) == 8);
    static_assert(get_recursive<11>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<12>(TestRecursionComplex{}) == 9);
    static_assert(get_recursive<13>(TestRecursionComplex{}) == 10);
    static_assert(get_recursive<14>(TestRecursionComplex{}) == 9);
    static_assert(get_recursive<15>(TestRecursionComplex{}) == 10);
    static_assert(get_recursive<16>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<17>(TestRecursionComplex{}) == 11);
    static_assert(get_recursive<18>(TestRecursionComplex{}) == 12);
    static_assert(get_recursive<19>(TestRecursionComplex{}) == 13);
    static_assert(get_recursive<20>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<21>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<22>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<23>(TestRecursionComplex{}) == 13);
    static_assert(get_recursive<24>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<25>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<26>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<27>(TestRecursionComplex{}) == 13);
    static_assert(get_recursive<28>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<29>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<30>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<31>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<32>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<33>(TestRecursionComplex{}) == nullptr);
  }

  Launcher(1, [] ARIA_DEVICE(int i) {
    TestRecursionComplex v;

    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<3>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<4>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<5>(v)), int64 &>);
    static_assert(std::is_same_v<decltype(get_recursive<6>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<7>(v)), int64 &>);
    static_assert(std::is_same_v<decltype(get_recursive<8>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<9>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<10>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<11>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<12>(v)), int64 &>);
    static_assert(std::is_same_v<decltype(get_recursive<13>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<14>(v)), int64 &>);
    static_assert(std::is_same_v<decltype(get_recursive<15>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<16>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<17>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<18>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<19>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<20>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<21>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<22>(v)), float *&>);
    static_assert(std::is_same_v<decltype(get_recursive<23>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<24>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<25>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<26>(v)), float *&>);
    static_assert(std::is_same_v<decltype(get_recursive<27>(v)), float &>);
    static_assert(std::is_same_v<decltype(get_recursive<28>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<29>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<30>(v)), float *&>);
    static_assert(std::is_same_v<decltype(get_recursive<31>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<32>(v)), double *&>);
    static_assert(std::is_same_v<decltype(get_recursive<33>(v)), float *&>);

    static_assert(std::is_same_v<decltype(get_recursive<0>(TestRecursionComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(TestRecursionComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(TestRecursionComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<3>(TestRecursionComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<4>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<5>(TestRecursionComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<6>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<7>(TestRecursionComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<8>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<9>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<10>(TestRecursionComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<11>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<12>(TestRecursionComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<13>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<14>(TestRecursionComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<15>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<16>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<17>(TestRecursionComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<18>(TestRecursionComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<19>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<20>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<21>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<22>(TestRecursionComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<23>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<24>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<25>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<26>(TestRecursionComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<27>(TestRecursionComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<28>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<29>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<30>(TestRecursionComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<31>(TestRecursionComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<32>(TestRecursionComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<33>(TestRecursionComplex{})), float *>);

    ARIA_ASSERT(get_recursive<0>(v) == 5);
    ARIA_ASSERT(get_recursive<1>(v) == 6);
    ARIA_ASSERT(get_recursive<2>(v) == 7);
    ARIA_ASSERT(get_recursive<3>(v) == 8);
    ARIA_ASSERT(get_recursive<4>(v) == nullptr);
    ARIA_ASSERT(get_recursive<5>(v) == 9);
    ARIA_ASSERT(get_recursive<6>(v) == 10);
    ARIA_ASSERT(get_recursive<7>(v) == 9);
    ARIA_ASSERT(get_recursive<8>(v) == 10);
    ARIA_ASSERT(get_recursive<9>(v) == nullptr);
    ARIA_ASSERT(get_recursive<10>(v) == 8);
    ARIA_ASSERT(get_recursive<11>(v) == nullptr);
    ARIA_ASSERT(get_recursive<12>(v) == 9);
    ARIA_ASSERT(get_recursive<13>(v) == 10);
    ARIA_ASSERT(get_recursive<14>(v) == 9);
    ARIA_ASSERT(get_recursive<15>(v) == 10);
    ARIA_ASSERT(get_recursive<16>(v) == nullptr);
    ARIA_ASSERT(get_recursive<17>(v) == 11);
    ARIA_ASSERT(get_recursive<18>(v) == 12);
    ARIA_ASSERT(get_recursive<19>(v) == 13);
    ARIA_ASSERT(get_recursive<20>(v) == nullptr);
    ARIA_ASSERT(get_recursive<21>(v) == nullptr);
    ARIA_ASSERT(get_recursive<22>(v) == nullptr);
    ARIA_ASSERT(get_recursive<23>(v) == 13);
    ARIA_ASSERT(get_recursive<24>(v) == nullptr);
    ARIA_ASSERT(get_recursive<25>(v) == nullptr);
    ARIA_ASSERT(get_recursive<26>(v) == nullptr);
    ARIA_ASSERT(get_recursive<27>(v) == 13);
    ARIA_ASSERT(get_recursive<28>(v) == nullptr);
    ARIA_ASSERT(get_recursive<29>(v) == nullptr);
    ARIA_ASSERT(get_recursive<30>(v) == nullptr);
    ARIA_ASSERT(get_recursive<31>(v) == nullptr);
    ARIA_ASSERT(get_recursive<32>(v) == nullptr);
    ARIA_ASSERT(get_recursive<33>(v) == nullptr);

    ARIA_ASSERT(get_recursive<0>(TestRecursionComplex{}) == 5);
    ARIA_ASSERT(get_recursive<1>(TestRecursionComplex{}) == 6);
    ARIA_ASSERT(get_recursive<2>(TestRecursionComplex{}) == 7);
    ARIA_ASSERT(get_recursive<3>(TestRecursionComplex{}) == 8);
    ARIA_ASSERT(get_recursive<4>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<5>(TestRecursionComplex{}) == 9);
    ARIA_ASSERT(get_recursive<6>(TestRecursionComplex{}) == 10);
    ARIA_ASSERT(get_recursive<7>(TestRecursionComplex{}) == 9);
    ARIA_ASSERT(get_recursive<8>(TestRecursionComplex{}) == 10);
    ARIA_ASSERT(get_recursive<9>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<10>(TestRecursionComplex{}) == 8);
    ARIA_ASSERT(get_recursive<11>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<12>(TestRecursionComplex{}) == 9);
    ARIA_ASSERT(get_recursive<13>(TestRecursionComplex{}) == 10);
    ARIA_ASSERT(get_recursive<14>(TestRecursionComplex{}) == 9);
    ARIA_ASSERT(get_recursive<15>(TestRecursionComplex{}) == 10);
    ARIA_ASSERT(get_recursive<16>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<17>(TestRecursionComplex{}) == 11);
    ARIA_ASSERT(get_recursive<18>(TestRecursionComplex{}) == 12);
    ARIA_ASSERT(get_recursive<19>(TestRecursionComplex{}) == 13);
    ARIA_ASSERT(get_recursive<20>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<21>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<22>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<23>(TestRecursionComplex{}) == 13);
    ARIA_ASSERT(get_recursive<24>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<25>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<26>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<27>(TestRecursionComplex{}) == 13);
    ARIA_ASSERT(get_recursive<28>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<29>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<30>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<31>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<32>(TestRecursionComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<33>(TestRecursionComplex{}) == nullptr);

    static_assert(get_recursive<0>(TestRecursionComplex{}) == 5);
    static_assert(get_recursive<1>(TestRecursionComplex{}) == 6);
    static_assert(get_recursive<2>(TestRecursionComplex{}) == 7);
    static_assert(get_recursive<3>(TestRecursionComplex{}) == 8);
    static_assert(get_recursive<4>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<5>(TestRecursionComplex{}) == 9);
    static_assert(get_recursive<6>(TestRecursionComplex{}) == 10);
    static_assert(get_recursive<7>(TestRecursionComplex{}) == 9);
    static_assert(get_recursive<8>(TestRecursionComplex{}) == 10);
    static_assert(get_recursive<9>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<10>(TestRecursionComplex{}) == 8);
    static_assert(get_recursive<11>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<12>(TestRecursionComplex{}) == 9);
    static_assert(get_recursive<13>(TestRecursionComplex{}) == 10);
    static_assert(get_recursive<14>(TestRecursionComplex{}) == 9);
    static_assert(get_recursive<15>(TestRecursionComplex{}) == 10);
    static_assert(get_recursive<16>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<17>(TestRecursionComplex{}) == 11);
    static_assert(get_recursive<18>(TestRecursionComplex{}) == 12);
    static_assert(get_recursive<19>(TestRecursionComplex{}) == 13);
    static_assert(get_recursive<20>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<21>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<22>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<23>(TestRecursionComplex{}) == 13);
    static_assert(get_recursive<24>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<25>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<26>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<27>(TestRecursionComplex{}) == 13);
    static_assert(get_recursive<28>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<29>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<30>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<31>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<32>(TestRecursionComplex{}) == nullptr);
    static_assert(get_recursive<33>(TestRecursionComplex{}) == nullptr);
  }).Launch();
  cuda::device::current::get().synchronize();
}

template <typename T>
class Vec3 {
public:
  Vec3(const T &x, const T &y, const T &z) : x_(x), y_(y), z_(z) {}

  ARIA_COPY_MOVE_ABILITY(Vec3, default, default);

  ARIA_REF_PROP(public, , x, x_);
  ARIA_REF_PROP(public, , y, y_);
  ARIA_REF_PROP(public, , z, z_);

private:
  T x_{}, y_{}, z_{};
};

template <typename T>
struct TestVec3 {
  T x, y, z;
};

} // namespace

template <>
struct Mosaic<float, float> {
  float operator()(float v) const {
    ARIA_ASSERT(!std::isnan(v));
    return v;
  }
};

template <>
struct Mosaic<double, float> {
  float operator()(double v) const { return v; }

  double operator()(float v) const { return v; }
};

template <>
struct Mosaic<float, double> {
  double operator()(float v) const { return v; }

  float operator()(double v) const { return v; }
};

template <typename T>
struct Mosaic<Vec3<T>, TestVec3<T>> {
  TestVec3<T> operator()(const Vec3<T> &v) const { return {.x = v.x(), .y = v.y(), .z = v.z()}; }

  Vec3<T> operator()(const TestVec3<T> &v) const { return {v.x, v.y, v.z}; }
};

TEST(Mosaic, Base) {
  using namespace mosaic::detail;

  // Non-pointer types.
  {
    static_assert(std::is_scalar_v<int>);
    static_assert(!std::is_aggregate_v<int>);
    static_assert(tuple_size_v<int> == 1);
    static_assert(MosaicPattern<int>);
    static_assert(tuple_size_recursive_v<int> == 1);
    static_assert(IRec2INonRec<0, int>() == 0);
    static_assert(IRec2INonRec<1, int>() == 1);
    static_assert(IRec2INonRec<99999, int>() == 1);
    static_assert(INonRec2IRec<0, int>() == 0);
    static_assert(INonRec2IRec<1, int>() == 1);
    static_assert(INonRec2IRec<99999, int>() == 1);
    static_assert(std::is_same_v<mosaic_pattern_types_recursive_t<int>, TypeArray<int>>);

    static_assert(!std::is_scalar_v<const int &>);
    static_assert(!std::is_aggregate_v<const int &>);
    // static_assert(tuple_size_v<const int&> == 1);
    static_assert(MosaicPattern<const int &>);
    static_assert(tuple_size_recursive_v<const int &> == 1);
    static_assert(IRec2INonRec<0, const int &>() == 0);
    static_assert(IRec2INonRec<1, const int &>() == 1);
    static_assert(IRec2INonRec<99999, const int &>() == 1);
    static_assert(INonRec2IRec<0, const int &>() == 0);
    static_assert(INonRec2IRec<1, const int &>() == 1);
    static_assert(INonRec2IRec<99999, const int &>() == 1);
    static_assert(std::is_same_v<mosaic_pattern_types_recursive_t<const int &>, TypeArray<int>>);

    static_assert(!std::is_scalar_v<int[3]>);
    static_assert(std::is_aggregate_v<int[3]>);
    static_assert(tuple_size_v<int[3]> == 3);
    static_assert(MosaicPattern<int[3]>);
    static_assert(tuple_size_recursive_v<int[3]> == 3);
    static_assert(IRec2INonRec<0, int[3]>() == 0);
    static_assert(IRec2INonRec<1, int[3]>() == 1);
    static_assert(IRec2INonRec<2, int[3]>() == 2);
    static_assert(IRec2INonRec<3, int[3]>() == 3);
    static_assert(IRec2INonRec<99999, int[3]>() == 3);
    static_assert(INonRec2IRec<0, int[3]>() == 0);
    static_assert(INonRec2IRec<1, int[3]>() == 1);
    static_assert(INonRec2IRec<2, int[3]>() == 2);
    static_assert(INonRec2IRec<3, int[3]>() == 3);
    static_assert(INonRec2IRec<99999, int[3]>() == 3);
    static_assert(std::is_same_v<mosaic_pattern_types_recursive_t<int[3]>, TypeArray<int, int, int>>);

    static_assert(!std::is_scalar_v<Test0Member>);
    static_assert(std::is_aggregate_v<Test0Member>);
    static_assert(tuple_size_v<Test0Member> == 0);
    static_assert(!MosaicPattern<Test0Member>);

    static_assert(!std::is_scalar_v<Test1Member>);
    static_assert(std::is_aggregate_v<Test1Member>);
    // static_assert(tuple_size_v<Test1Member> == 1);
    // static_assert(!MosaicPattern<Test1Member>);

    static_assert(!std::is_scalar_v<Test1ArrayMember>);
    static_assert(std::is_aggregate_v<Test1ArrayMember>);
    static_assert(tuple_size_v<Test1ArrayMember> == 1);
    static_assert(MosaicPattern<Test1ArrayMember>);
    static_assert(tuple_size_recursive_v<Test1ArrayMember> == 3);
    static_assert(IRec2INonRec<0, Test1ArrayMember>() == 0);
    static_assert(IRec2INonRec<1, Test1ArrayMember>() == 0);
    static_assert(IRec2INonRec<2, Test1ArrayMember>() == 0);
    static_assert(IRec2INonRec<3, Test1ArrayMember>() == 1);
    static_assert(IRec2INonRec<99999, Test1ArrayMember>() == 1);
    static_assert(INonRec2IRec<0, Test1ArrayMember>() == 0);
    static_assert(INonRec2IRec<1, Test1ArrayMember>() == 3);
    static_assert(INonRec2IRec<99999, Test1ArrayMember>() == 3);
    static_assert(std::is_same_v<mosaic_pattern_types_recursive_t<Test1ArrayMember>, TypeArray<int, int, int>>);

    static_assert(!std::is_scalar_v<Test2Members>);
    static_assert(std::is_aggregate_v<Test2Members>);
    static_assert(tuple_size_v<Test2Members> == 2);
    static_assert(MosaicPattern<Test2Members>);
    static_assert(tuple_size_recursive_v<Test2Members> == 2);
    static_assert(IRec2INonRec<0, Test2Members>() == 0);
    static_assert(IRec2INonRec<1, Test2Members>() == 1);
    static_assert(IRec2INonRec<2, Test2Members>() == 2);
    static_assert(IRec2INonRec<99999, Test2Members>() == 2);
    static_assert(INonRec2IRec<0, Test2Members>() == 0);
    static_assert(INonRec2IRec<1, Test2Members>() == 1);
    static_assert(INonRec2IRec<2, Test2Members>() == 2);
    static_assert(INonRec2IRec<99999, Test2Members>() == 2);
    static_assert(std::is_same_v<mosaic_pattern_types_recursive_t<Test2Members>, TypeArray<int, double>>);

    static_assert(!std::is_scalar_v<TestPrivateMembers>);
    static_assert(!std::is_aggregate_v<TestPrivateMembers>);
    // static_assert(tuple_size_v<TestPrivateMembers> == 1);
    // static_assert(!MosaicPattern<TestPrivateMembers>);

    static_assert(!std::is_scalar_v<TestInheritance>);
    static_assert(std::is_aggregate_v<TestInheritance>);
    // static_assert(tuple_size_v<TestInheritance> == 2);
    // static_assert(!MosaicPattern<TestInheritance>);

    static_assert(!std::is_scalar_v<TestRecursion0Member>);
    static_assert(std::is_aggregate_v<TestRecursion0Member>);
    // static_assert(tuple_size_v<TestRecursion0Member> == 2);
    // static_assert(!MosaicPattern<TestRecursion0Member>);

    static_assert(!std::is_scalar_v<TestRecursion1Member>);
    static_assert(std::is_aggregate_v<TestRecursion1Member>);
    static_assert(tuple_size_v<TestRecursion1Member> == 2);
    // static_assert(!MosaicPattern<TestRecursion1Member>);

    static_assert(!std::is_scalar_v<TestRecursion2Members>);
    static_assert(std::is_aggregate_v<TestRecursion2Members>);
    static_assert(tuple_size_v<TestRecursion2Members> == 2);
    static_assert(MosaicPattern<TestRecursion2Members>);
    static_assert(tuple_size_recursive_v<TestRecursion2Members> == 3);
    static_assert(IRec2INonRec<0, TestRecursion2Members>() == 0);
    static_assert(IRec2INonRec<1, TestRecursion2Members>() == 1);
    static_assert(IRec2INonRec<2, TestRecursion2Members>() == 1);
    static_assert(IRec2INonRec<3, TestRecursion2Members>() == 2);
    static_assert(IRec2INonRec<99999, TestRecursion2Members>() == 2);
    static_assert(INonRec2IRec<0, TestRecursion2Members>() == 0);
    static_assert(INonRec2IRec<1, TestRecursion2Members>() == 1);
    static_assert(INonRec2IRec<2, TestRecursion2Members>() == 3);
    static_assert(INonRec2IRec<99999, TestRecursion2Members>() == 3);
    static_assert(
        std::is_same_v<mosaic_pattern_types_recursive_t<TestRecursion2Members>, TypeArray<int, double, int *>>);

    static_assert(!std::is_scalar_v<TestRecursionComplex>);
    static_assert(std::is_aggregate_v<TestRecursionComplex>);
    static_assert(tuple_size_v<TestRecursionComplex> == 13);
    static_assert(MosaicPattern<TestRecursionComplex>);
    static_assert(tuple_size_recursive_v<TestRecursionComplex> == 34);
    static_assert(IRec2INonRec<0, TestRecursionComplex>() == 0);
    static_assert(IRec2INonRec<1, TestRecursionComplex>() == 1);
    static_assert(IRec2INonRec<2, TestRecursionComplex>() == 2);
    ForEach<7>([&]<auto i>() { static_assert(IRec2INonRec<3 + i, TestRecursionComplex>() == 3); });
    ForEach<7>([&]<auto i>() { static_assert(IRec2INonRec<10 + i, TestRecursionComplex>() == 4); });
    static_assert(IRec2INonRec<17, TestRecursionComplex>() == 5);
    static_assert(IRec2INonRec<18, TestRecursionComplex>() == 6);
    ForEach<4>([&]<auto i>() { static_assert(IRec2INonRec<19 + i, TestRecursionComplex>() == 7); });
    ForEach<4>([&]<auto i>() { static_assert(IRec2INonRec<23 + i, TestRecursionComplex>() == 8); });
    ForEach<4>([&]<auto i>() { static_assert(IRec2INonRec<27 + i, TestRecursionComplex>() == 9); });
    static_assert(IRec2INonRec<31, TestRecursionComplex>() == 10);
    static_assert(IRec2INonRec<32, TestRecursionComplex>() == 11);
    static_assert(IRec2INonRec<33, TestRecursionComplex>() == 12);
    static_assert(IRec2INonRec<34, TestRecursionComplex>() == 13);
    static_assert(IRec2INonRec<99999, TestRecursionComplex>() == 13);
    static_assert(INonRec2IRec<0, TestRecursionComplex>() == 0);
    static_assert(INonRec2IRec<1, TestRecursionComplex>() == 1);
    static_assert(INonRec2IRec<2, TestRecursionComplex>() == 2);
    static_assert(INonRec2IRec<3, TestRecursionComplex>() == 3);
    static_assert(INonRec2IRec<4, TestRecursionComplex>() == 10);
    static_assert(INonRec2IRec<5, TestRecursionComplex>() == 17);
    static_assert(INonRec2IRec<6, TestRecursionComplex>() == 18);
    static_assert(INonRec2IRec<7, TestRecursionComplex>() == 19);
    static_assert(INonRec2IRec<8, TestRecursionComplex>() == 23);
    static_assert(INonRec2IRec<9, TestRecursionComplex>() == 27);
    static_assert(INonRec2IRec<10, TestRecursionComplex>() == 31);
    static_assert(INonRec2IRec<11, TestRecursionComplex>() == 32);
    static_assert(INonRec2IRec<12, TestRecursionComplex>() == 33);
    static_assert(INonRec2IRec<13, TestRecursionComplex>() == 34);
    static_assert(INonRec2IRec<99999, TestRecursionComplex>() == 34);
    static_assert(std::is_same_v<
                  mosaic_pattern_types_recursive_t<TestRecursionComplex>,
                  TypeArray<int, int, int, double, int *, int64, float, int64, float, double *, double, int *, int64,
                            float, int64, float, double *, double, double, float, int *, double *, float *, float,
                            int *, double *, float *, float, int *, double *, float *, int *, double *, float *>>);
  }

  // Pointer types.
  {
    auto testPointerType = []<typename T>() {
      static_assert(!std::is_pointer_v<T>);

      static_assert(std::is_scalar_v<T *>);
      static_assert(!std::is_aggregate_v<T *>);
      static_assert(tuple_size_v<T *> == 1);
      static_assert(MosaicPattern<T *>);
      static_assert(tuple_size_recursive_v<T *> == 1);
      static_assert(IRec2INonRec<0, T *>() == 0);
      static_assert(IRec2INonRec<1, T *>() == 1);
      static_assert(IRec2INonRec<99999, T *>() == 1);
      static_assert(std::is_same_v<mosaic_pattern_types_recursive_t<T *>, TypeArray<T *>>);
    };

    testPointerType.operator()<double>();
    testPointerType.operator()<Test0Member>();
    testPointerType.operator()<Test1Member>();
    testPointerType.operator()<Test1ArrayMember>();
    testPointerType.operator()<Test2Members>();
    testPointerType.operator()<TestPrivateMembers>();
    testPointerType.operator()<TestInheritance>();
    testPointerType.operator()<TestRecursion0Member>();
    testPointerType.operator()<TestRecursion1Member>();
    testPointerType.operator()<TestRecursion2Members>();
    testPointerType.operator()<TestRecursionComplex>();
  }
}

TEST(Mosaic, GetRecursive) {
  TestGetRecursive();
}

TEST(Mosaic, ValidMoasic) {
  using namespace mosaic::detail;

  static_assert(ValidMosaic<Mosaic<float, float>>);
  static_assert(ValidMosaic<Mosaic<double, float>>);
  static_assert(ValidMosaic<Mosaic<float, double>>);

  static_assert(ValidMosaic<Mosaic<Vec3<int>, TestVec3<int>>>);
  static_assert(ValidMosaic<Mosaic<Vec3<double>, TestVec3<double>>>);
  static_assert(ValidMosaic<Mosaic<Vec3<int *>, TestVec3<int *>>>);
  static_assert(ValidMosaic<Mosaic<Vec3<double *>, TestVec3<double *>>>);
  // static_assert(ValidMosaic<Mosaic<Vec3<std::string>, TestVec3<std::string>>>);
}

} // namespace ARIA
