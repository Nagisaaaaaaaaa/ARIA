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
  int v0 = 5, v1 = 5, v2 = 5;

  struct {
    double vs0 = 6;
    int *vs1 = nullptr;

    struct {
      int vss0 = 7;
      double vss1 = 8;
    } ss0, ss1;

    double *vs2 = nullptr;
  } s0, s1;

  double v3 = 9, v4 = 9;

  struct {
    double vs0 = 10;
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
