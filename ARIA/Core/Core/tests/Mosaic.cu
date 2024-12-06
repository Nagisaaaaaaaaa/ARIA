#include "ARIA/Launcher.h"
#include "ARIA/Mosaic.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct Test0Member {};

//! Structures with 1 member is considered unnecessary.
struct Test1Member {
  int v = 5;
};

struct Test2Members {
  int v0 = 5;
  double v1 = 6;
};

struct TestPrivateMembers {
private:
  int v_ = 5;
};

struct TestRecursive0Member {
  int v0 = 5;

  struct {
  } s0;
};

//! Structures with 1 member is considered unnecessary.
struct TestRecursive1Member {
  int v0 = 5;

  struct {
    double v1 = 6;
  } s0;
};

struct TestRecursive2Members {
  int v0 = 5;

  struct {
    double v1 = 6;
    int *v2 = nullptr;
  } s0;
};

struct TestRecursiveComplex {
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
    TestRecursive2Members v;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(TestRecursive2Members{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(TestRecursive2Members{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(TestRecursive2Members{})), int *>);
    EXPECT_EQ(get_recursive<0>(v), 5);
    EXPECT_EQ(get_recursive<1>(v), 6);
    EXPECT_EQ(get_recursive<2>(v), nullptr);
    EXPECT_EQ(get_recursive<0>(TestRecursive2Members{}), 5);
    EXPECT_EQ(get_recursive<1>(TestRecursive2Members{}), 6);
    EXPECT_EQ(get_recursive<2>(TestRecursive2Members{}), nullptr);
    static_assert(get_recursive<0>(TestRecursive2Members{}) == 5);
    static_assert(get_recursive<1>(TestRecursive2Members{}) == 6);
    static_assert(get_recursive<2>(TestRecursive2Members{}) == nullptr);
  }

  Launcher(1, [] ARIA_DEVICE(int i) {
    TestRecursive2Members v;
    static_assert(std::is_same_v<decltype(get_recursive<0>(v)), int &>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(v)), double &>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(v)), int *&>);
    static_assert(std::is_same_v<decltype(get_recursive<0>(TestRecursive2Members{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(TestRecursive2Members{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(TestRecursive2Members{})), int *>);
    ARIA_ASSERT(get_recursive<0>(v) == 5);
    ARIA_ASSERT(get_recursive<1>(v) == 6);
    ARIA_ASSERT(get_recursive<2>(v) == nullptr);
    ARIA_ASSERT(get_recursive<0>(TestRecursive2Members{}) == 5);
    ARIA_ASSERT(get_recursive<1>(TestRecursive2Members{}) == 6);
    ARIA_ASSERT(get_recursive<2>(TestRecursive2Members{}) == nullptr);
    static_assert(get_recursive<0>(TestRecursive2Members{}) == 5);
    static_assert(get_recursive<1>(TestRecursive2Members{}) == 6);
    static_assert(get_recursive<2>(TestRecursive2Members{}) == nullptr);
  }).Launch();
  cuda::device::current::get().synchronize();

  {
    TestRecursiveComplex v;

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

    static_assert(std::is_same_v<decltype(get_recursive<0>(TestRecursiveComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(TestRecursiveComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(TestRecursiveComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<3>(TestRecursiveComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<4>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<5>(TestRecursiveComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<6>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<7>(TestRecursiveComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<8>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<9>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<10>(TestRecursiveComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<11>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<12>(TestRecursiveComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<13>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<14>(TestRecursiveComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<15>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<16>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<17>(TestRecursiveComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<18>(TestRecursiveComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<19>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<20>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<21>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<22>(TestRecursiveComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<23>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<24>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<25>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<26>(TestRecursiveComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<27>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<28>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<29>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<30>(TestRecursiveComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<31>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<32>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<33>(TestRecursiveComplex{})), float *>);

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

    EXPECT_EQ(get_recursive<0>(TestRecursiveComplex{}), 5);
    EXPECT_EQ(get_recursive<1>(TestRecursiveComplex{}), 6);
    EXPECT_EQ(get_recursive<2>(TestRecursiveComplex{}), 7);
    EXPECT_EQ(get_recursive<3>(TestRecursiveComplex{}), 8);
    EXPECT_EQ(get_recursive<4>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<5>(TestRecursiveComplex{}), 9);
    EXPECT_EQ(get_recursive<6>(TestRecursiveComplex{}), 10);
    EXPECT_EQ(get_recursive<7>(TestRecursiveComplex{}), 9);
    EXPECT_EQ(get_recursive<8>(TestRecursiveComplex{}), 10);
    EXPECT_EQ(get_recursive<9>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<10>(TestRecursiveComplex{}), 8);
    EXPECT_EQ(get_recursive<11>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<12>(TestRecursiveComplex{}), 9);
    EXPECT_EQ(get_recursive<13>(TestRecursiveComplex{}), 10);
    EXPECT_EQ(get_recursive<14>(TestRecursiveComplex{}), 9);
    EXPECT_EQ(get_recursive<15>(TestRecursiveComplex{}), 10);
    EXPECT_EQ(get_recursive<16>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<17>(TestRecursiveComplex{}), 11);
    EXPECT_EQ(get_recursive<18>(TestRecursiveComplex{}), 12);
    EXPECT_EQ(get_recursive<19>(TestRecursiveComplex{}), 13);
    EXPECT_EQ(get_recursive<20>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<21>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<22>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<23>(TestRecursiveComplex{}), 13);
    EXPECT_EQ(get_recursive<24>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<25>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<26>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<27>(TestRecursiveComplex{}), 13);
    EXPECT_EQ(get_recursive<28>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<29>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<30>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<31>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<32>(TestRecursiveComplex{}), nullptr);
    EXPECT_EQ(get_recursive<33>(TestRecursiveComplex{}), nullptr);

    static_assert(get_recursive<0>(TestRecursiveComplex{}) == 5);
    static_assert(get_recursive<1>(TestRecursiveComplex{}) == 6);
    static_assert(get_recursive<2>(TestRecursiveComplex{}) == 7);
    static_assert(get_recursive<3>(TestRecursiveComplex{}) == 8);
    static_assert(get_recursive<4>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<5>(TestRecursiveComplex{}) == 9);
    static_assert(get_recursive<6>(TestRecursiveComplex{}) == 10);
    static_assert(get_recursive<7>(TestRecursiveComplex{}) == 9);
    static_assert(get_recursive<8>(TestRecursiveComplex{}) == 10);
    static_assert(get_recursive<9>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<10>(TestRecursiveComplex{}) == 8);
    static_assert(get_recursive<11>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<12>(TestRecursiveComplex{}) == 9);
    static_assert(get_recursive<13>(TestRecursiveComplex{}) == 10);
    static_assert(get_recursive<14>(TestRecursiveComplex{}) == 9);
    static_assert(get_recursive<15>(TestRecursiveComplex{}) == 10);
    static_assert(get_recursive<16>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<17>(TestRecursiveComplex{}) == 11);
    static_assert(get_recursive<18>(TestRecursiveComplex{}) == 12);
    static_assert(get_recursive<19>(TestRecursiveComplex{}) == 13);
    static_assert(get_recursive<20>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<21>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<22>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<23>(TestRecursiveComplex{}) == 13);
    static_assert(get_recursive<24>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<25>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<26>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<27>(TestRecursiveComplex{}) == 13);
    static_assert(get_recursive<28>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<29>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<30>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<31>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<32>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<33>(TestRecursiveComplex{}) == nullptr);
  }

  Launcher(1, [] ARIA_DEVICE(int i) {
    TestRecursiveComplex v;

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

    static_assert(std::is_same_v<decltype(get_recursive<0>(TestRecursiveComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<1>(TestRecursiveComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<2>(TestRecursiveComplex{})), int>);
    static_assert(std::is_same_v<decltype(get_recursive<3>(TestRecursiveComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<4>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<5>(TestRecursiveComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<6>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<7>(TestRecursiveComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<8>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<9>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<10>(TestRecursiveComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<11>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<12>(TestRecursiveComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<13>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<14>(TestRecursiveComplex{})), int64>);
    static_assert(std::is_same_v<decltype(get_recursive<15>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<16>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<17>(TestRecursiveComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<18>(TestRecursiveComplex{})), double>);
    static_assert(std::is_same_v<decltype(get_recursive<19>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<20>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<21>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<22>(TestRecursiveComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<23>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<24>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<25>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<26>(TestRecursiveComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<27>(TestRecursiveComplex{})), float>);
    static_assert(std::is_same_v<decltype(get_recursive<28>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<29>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<30>(TestRecursiveComplex{})), float *>);
    static_assert(std::is_same_v<decltype(get_recursive<31>(TestRecursiveComplex{})), int *>);
    static_assert(std::is_same_v<decltype(get_recursive<32>(TestRecursiveComplex{})), double *>);
    static_assert(std::is_same_v<decltype(get_recursive<33>(TestRecursiveComplex{})), float *>);

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

    ARIA_ASSERT(get_recursive<0>(TestRecursiveComplex{}) == 5);
    ARIA_ASSERT(get_recursive<1>(TestRecursiveComplex{}) == 6);
    ARIA_ASSERT(get_recursive<2>(TestRecursiveComplex{}) == 7);
    ARIA_ASSERT(get_recursive<3>(TestRecursiveComplex{}) == 8);
    ARIA_ASSERT(get_recursive<4>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<5>(TestRecursiveComplex{}) == 9);
    ARIA_ASSERT(get_recursive<6>(TestRecursiveComplex{}) == 10);
    ARIA_ASSERT(get_recursive<7>(TestRecursiveComplex{}) == 9);
    ARIA_ASSERT(get_recursive<8>(TestRecursiveComplex{}) == 10);
    ARIA_ASSERT(get_recursive<9>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<10>(TestRecursiveComplex{}) == 8);
    ARIA_ASSERT(get_recursive<11>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<12>(TestRecursiveComplex{}) == 9);
    ARIA_ASSERT(get_recursive<13>(TestRecursiveComplex{}) == 10);
    ARIA_ASSERT(get_recursive<14>(TestRecursiveComplex{}) == 9);
    ARIA_ASSERT(get_recursive<15>(TestRecursiveComplex{}) == 10);
    ARIA_ASSERT(get_recursive<16>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<17>(TestRecursiveComplex{}) == 11);
    ARIA_ASSERT(get_recursive<18>(TestRecursiveComplex{}) == 12);
    ARIA_ASSERT(get_recursive<19>(TestRecursiveComplex{}) == 13);
    ARIA_ASSERT(get_recursive<20>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<21>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<22>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<23>(TestRecursiveComplex{}) == 13);
    ARIA_ASSERT(get_recursive<24>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<25>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<26>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<27>(TestRecursiveComplex{}) == 13);
    ARIA_ASSERT(get_recursive<28>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<29>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<30>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<31>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<32>(TestRecursiveComplex{}) == nullptr);
    ARIA_ASSERT(get_recursive<33>(TestRecursiveComplex{}) == nullptr);

    static_assert(get_recursive<0>(TestRecursiveComplex{}) == 5);
    static_assert(get_recursive<1>(TestRecursiveComplex{}) == 6);
    static_assert(get_recursive<2>(TestRecursiveComplex{}) == 7);
    static_assert(get_recursive<3>(TestRecursiveComplex{}) == 8);
    static_assert(get_recursive<4>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<5>(TestRecursiveComplex{}) == 9);
    static_assert(get_recursive<6>(TestRecursiveComplex{}) == 10);
    static_assert(get_recursive<7>(TestRecursiveComplex{}) == 9);
    static_assert(get_recursive<8>(TestRecursiveComplex{}) == 10);
    static_assert(get_recursive<9>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<10>(TestRecursiveComplex{}) == 8);
    static_assert(get_recursive<11>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<12>(TestRecursiveComplex{}) == 9);
    static_assert(get_recursive<13>(TestRecursiveComplex{}) == 10);
    static_assert(get_recursive<14>(TestRecursiveComplex{}) == 9);
    static_assert(get_recursive<15>(TestRecursiveComplex{}) == 10);
    static_assert(get_recursive<16>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<17>(TestRecursiveComplex{}) == 11);
    static_assert(get_recursive<18>(TestRecursiveComplex{}) == 12);
    static_assert(get_recursive<19>(TestRecursiveComplex{}) == 13);
    static_assert(get_recursive<20>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<21>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<22>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<23>(TestRecursiveComplex{}) == 13);
    static_assert(get_recursive<24>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<25>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<26>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<27>(TestRecursiveComplex{}) == 13);
    static_assert(get_recursive<28>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<29>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<30>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<31>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<32>(TestRecursiveComplex{}) == nullptr);
    static_assert(get_recursive<33>(TestRecursiveComplex{}) == nullptr);
  }).Launch();
  cuda::device::current::get().synchronize();
}

} // namespace

TEST(Mosaic, Base) {
  // Non-pointer types.
  {
    static_assert(std::is_scalar_v<int>);
    static_assert(!std::is_aggregate_v<int>);
    static_assert(boost::pfr::tuple_size_v<int> == 1);
    static_assert(MosaicPattern<int>);
    static_assert(tuple_size_recursive_v<int> == 1);
    static_assert(IRec2INonRec<0, int>() == 0);
    static_assert(IRec2INonRec<1, int>() == 1);
    static_assert(IRec2INonRec<99999, int>() == 1);
    static_assert(INonRec2IRec<0, int>() == 0);
    static_assert(INonRec2IRec<1, int>() == 1);
    static_assert(INonRec2IRec<99999, int>() == 1);

    static_assert(!std::is_scalar_v<Test0Member>);
    static_assert(std::is_aggregate_v<Test0Member>);
    static_assert(boost::pfr::tuple_size_v<Test0Member> == 0);
    static_assert(!MosaicPattern<Test0Member>);

    static_assert(!std::is_scalar_v<Test1Member>);
    static_assert(std::is_aggregate_v<Test1Member>);
    // static_assert(boost::pfr::tuple_size_v<Test1Member> == 1);
    // static_assert(!MosaicPattern<Test1Member>);

    static_assert(!std::is_scalar_v<Test2Members>);
    static_assert(std::is_aggregate_v<Test2Members>);
    static_assert(boost::pfr::tuple_size_v<Test2Members> == 2);
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

    static_assert(!std::is_scalar_v<TestPrivateMembers>);
    static_assert(!std::is_aggregate_v<TestPrivateMembers>);
    // static_assert(boost::pfr::tuple_size_v<TestPrivateMembers> == 0);
    // static_assert(!MosaicPattern<TestPrivateMembers>);

    static_assert(!std::is_scalar_v<TestRecursive0Member>);
    static_assert(std::is_aggregate_v<TestRecursive0Member>);
    static_assert(boost::pfr::tuple_size_v<TestRecursive0Member> == 2);
    static_assert(!MosaicPattern<TestRecursive0Member>);

    static_assert(!std::is_scalar_v<TestRecursive1Member>);
    static_assert(std::is_aggregate_v<TestRecursive1Member>);
    static_assert(boost::pfr::tuple_size_v<TestRecursive1Member> == 2);
    // static_assert(!MosaicPattern<TestRecursive1Member>);

    static_assert(!std::is_scalar_v<TestRecursive2Members>);
    static_assert(std::is_aggregate_v<TestRecursive2Members>);
    static_assert(boost::pfr::tuple_size_v<TestRecursive2Members> == 2);
    static_assert(MosaicPattern<TestRecursive2Members>);
    static_assert(tuple_size_recursive_v<TestRecursive2Members> == 3);
    static_assert(IRec2INonRec<0, TestRecursive2Members>() == 0);
    static_assert(IRec2INonRec<1, TestRecursive2Members>() == 1);
    static_assert(IRec2INonRec<2, TestRecursive2Members>() == 1);
    static_assert(IRec2INonRec<3, TestRecursive2Members>() == 2);
    static_assert(IRec2INonRec<99999, TestRecursive2Members>() == 2);
    static_assert(INonRec2IRec<0, TestRecursive2Members>() == 0);
    static_assert(INonRec2IRec<1, TestRecursive2Members>() == 1);
    static_assert(INonRec2IRec<2, TestRecursive2Members>() == 3);
    static_assert(INonRec2IRec<99999, TestRecursive2Members>() == 3);

    static_assert(!std::is_scalar_v<TestRecursiveComplex>);
    static_assert(std::is_aggregate_v<TestRecursiveComplex>);
    static_assert(boost::pfr::tuple_size_v<TestRecursiveComplex> == 13);
    static_assert(MosaicPattern<TestRecursiveComplex>);
    static_assert(tuple_size_recursive_v<TestRecursiveComplex> == 34);
    static_assert(IRec2INonRec<0, TestRecursiveComplex>() == 0);
    static_assert(IRec2INonRec<1, TestRecursiveComplex>() == 1);
    static_assert(IRec2INonRec<2, TestRecursiveComplex>() == 2);
    ForEach<7>([&]<auto i>() { static_assert(IRec2INonRec<3 + i, TestRecursiveComplex>() == 3); });
    ForEach<7>([&]<auto i>() { static_assert(IRec2INonRec<10 + i, TestRecursiveComplex>() == 4); });
    static_assert(IRec2INonRec<17, TestRecursiveComplex>() == 5);
    static_assert(IRec2INonRec<18, TestRecursiveComplex>() == 6);
    ForEach<4>([&]<auto i>() { static_assert(IRec2INonRec<19 + i, TestRecursiveComplex>() == 7); });
    ForEach<4>([&]<auto i>() { static_assert(IRec2INonRec<23 + i, TestRecursiveComplex>() == 8); });
    ForEach<4>([&]<auto i>() { static_assert(IRec2INonRec<27 + i, TestRecursiveComplex>() == 9); });
    static_assert(IRec2INonRec<31, TestRecursiveComplex>() == 10);
    static_assert(IRec2INonRec<32, TestRecursiveComplex>() == 11);
    static_assert(IRec2INonRec<33, TestRecursiveComplex>() == 12);
    static_assert(IRec2INonRec<34, TestRecursiveComplex>() == 13);
    static_assert(IRec2INonRec<99999, TestRecursiveComplex>() == 13);
    static_assert(INonRec2IRec<0, TestRecursiveComplex>() == 0);
    static_assert(INonRec2IRec<1, TestRecursiveComplex>() == 1);
    static_assert(INonRec2IRec<2, TestRecursiveComplex>() == 2);
    static_assert(INonRec2IRec<3, TestRecursiveComplex>() == 3);
    static_assert(INonRec2IRec<4, TestRecursiveComplex>() == 10);
    static_assert(INonRec2IRec<5, TestRecursiveComplex>() == 17);
    static_assert(INonRec2IRec<6, TestRecursiveComplex>() == 18);
    static_assert(INonRec2IRec<7, TestRecursiveComplex>() == 19);
    static_assert(INonRec2IRec<8, TestRecursiveComplex>() == 23);
    static_assert(INonRec2IRec<9, TestRecursiveComplex>() == 27);
    static_assert(INonRec2IRec<10, TestRecursiveComplex>() == 31);
    static_assert(INonRec2IRec<11, TestRecursiveComplex>() == 32);
    static_assert(INonRec2IRec<12, TestRecursiveComplex>() == 33);
    static_assert(INonRec2IRec<13, TestRecursiveComplex>() == 34);
    static_assert(INonRec2IRec<99999, TestRecursiveComplex>() == 34);
  }

  // Pointer types.
  {
    auto testPointerType = []<typename T>() {
      static_assert(!std::is_pointer_v<T>);

      static_assert(std::is_scalar_v<T *>);
      static_assert(!std::is_aggregate_v<T *>);
      static_assert(boost::pfr::tuple_size_v<T *> == 1);
      static_assert(MosaicPattern<T *>);
      static_assert(tuple_size_recursive_v<T *> == 1);
      static_assert(IRec2INonRec<0, T *>() == 0);
      static_assert(IRec2INonRec<1, T *>() == 1);
      static_assert(IRec2INonRec<99999, T *>() == 1);
    };

    testPointerType.operator()<double>();
    testPointerType.operator()<Test0Member>();
    testPointerType.operator()<Test1Member>();
    testPointerType.operator()<Test2Members>();
    testPointerType.operator()<TestPrivateMembers>();
    testPointerType.operator()<TestRecursive0Member>();
    testPointerType.operator()<TestRecursive1Member>();
    testPointerType.operator()<TestRecursive2Members>();
    testPointerType.operator()<TestRecursiveComplex>();
  }
}

TEST(Mosaic, GetRecursive) {
  TestGetRecursive();
}

} // namespace ARIA
