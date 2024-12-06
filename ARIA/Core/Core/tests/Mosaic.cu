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
  }

  // Pointer types.
  {
    static_assert(std::is_scalar_v<double *>);
    static_assert(!std::is_aggregate_v<double *>);
    static_assert(boost::pfr::tuple_size_v<double *> == 1);
    static_assert(MosaicPattern<double *>);
    static_assert(tuple_size_recursive_v<double *> == 1);

    static_assert(std::is_scalar_v<Test0Member *>);
    static_assert(!std::is_aggregate_v<Test0Member *>);
    static_assert(boost::pfr::tuple_size_v<Test0Member *> == 1);
    static_assert(MosaicPattern<Test0Member *>);
    static_assert(tuple_size_recursive_v<Test0Member *> == 1);

    static_assert(std::is_scalar_v<Test1Member *>);
    static_assert(!std::is_aggregate_v<Test1Member *>);
    static_assert(boost::pfr::tuple_size_v<Test1Member *> == 1);
    static_assert(MosaicPattern<Test1Member *>);
    static_assert(tuple_size_recursive_v<Test1Member *> == 1);

    static_assert(std::is_scalar_v<Test2Members *>);
    static_assert(!std::is_aggregate_v<Test2Members *>);
    static_assert(boost::pfr::tuple_size_v<Test2Members *> == 1);
    static_assert(MosaicPattern<Test2Members *>);
    static_assert(tuple_size_recursive_v<Test2Members *> == 1);

    static_assert(std::is_scalar_v<TestPrivateMembers *>);
    static_assert(!std::is_aggregate_v<TestPrivateMembers *>);
    static_assert(boost::pfr::tuple_size_v<TestPrivateMembers *> == 1);
    static_assert(MosaicPattern<TestPrivateMembers *>);
    static_assert(tuple_size_recursive_v<TestPrivateMembers *> == 1);

    static_assert(std::is_scalar_v<TestRecursive0Member *>);
    static_assert(!std::is_aggregate_v<TestRecursive0Member *>);
    static_assert(boost::pfr::tuple_size_v<TestRecursive0Member *> == 1);
    static_assert(MosaicPattern<TestRecursive0Member *>);
    static_assert(tuple_size_recursive_v<TestRecursive0Member *> == 1);

    static_assert(std::is_scalar_v<TestRecursive1Member *>);
    static_assert(!std::is_aggregate_v<TestRecursive1Member *>);
    static_assert(boost::pfr::tuple_size_v<TestRecursive1Member *> == 1);
    static_assert(MosaicPattern<TestRecursive1Member *>);
    static_assert(tuple_size_recursive_v<TestRecursive1Member *> == 1);

    static_assert(std::is_scalar_v<TestRecursive2Members *>);
    static_assert(!std::is_aggregate_v<TestRecursive2Members *>);
    static_assert(boost::pfr::tuple_size_v<TestRecursive2Members *> == 1);
    static_assert(MosaicPattern<TestRecursive2Members *>);
    static_assert(tuple_size_recursive_v<TestRecursive2Members *> == 1);

    static_assert(std::is_scalar_v<TestRecursiveComplex *>);
    static_assert(!std::is_aggregate_v<TestRecursiveComplex *>);
    static_assert(boost::pfr::tuple_size_v<TestRecursiveComplex *> == 1);
    static_assert(MosaicPattern<TestRecursiveComplex *>);
    static_assert(tuple_size_recursive_v<TestRecursiveComplex *> == 1);
  }
}

} // namespace ARIA
