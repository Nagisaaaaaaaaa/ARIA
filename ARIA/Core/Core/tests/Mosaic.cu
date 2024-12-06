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
  } s1;
};

//! Structures with 1 member is considered unnecessary.
struct TestRecursive1Member {
  int v0 = 5;

  struct {
    double v1 = 6;
  } s1;
};

struct TestRecursive2Members {
  int v0 = 5;

  struct {
    double v1 = 6;
    int *v2 = nullptr;
  } s1;
};

} // namespace

TEST(Mosaic, Base) {
  // Non-pointer types.
  {
    static_assert(std::is_scalar_v<int>);
    static_assert(!std::is_aggregate_v<int>);
    static_assert(boost::pfr::tuple_size_v<int> == 1);
    static_assert(MosaicPattern<int>);

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
  }

  // Pointer types.
  {
    static_assert(std::is_scalar_v<double *>);
    static_assert(!std::is_aggregate_v<double *>);
    static_assert(boost::pfr::tuple_size_v<double *> == 1);
    static_assert(MosaicPattern<double *>);

    static_assert(std::is_scalar_v<Test0Member *>);
    static_assert(!std::is_aggregate_v<Test0Member *>);
    static_assert(boost::pfr::tuple_size_v<Test0Member *> == 1);
    static_assert(MosaicPattern<Test0Member *>);

    static_assert(std::is_scalar_v<Test1Member *>);
    static_assert(!std::is_aggregate_v<Test1Member *>);
    static_assert(boost::pfr::tuple_size_v<Test1Member *> == 1);
    static_assert(MosaicPattern<Test1Member *>);

    static_assert(std::is_scalar_v<Test2Members *>);
    static_assert(!std::is_aggregate_v<Test2Members *>);
    static_assert(boost::pfr::tuple_size_v<Test2Members *> == 1);
    static_assert(MosaicPattern<Test2Members *>);

    static_assert(std::is_scalar_v<TestPrivateMembers *>);
    static_assert(!std::is_aggregate_v<TestPrivateMembers *>);
    static_assert(boost::pfr::tuple_size_v<TestPrivateMembers *> == 1);
    static_assert(MosaicPattern<TestPrivateMembers *>);

    static_assert(std::is_scalar_v<TestRecursive0Member *>);
    static_assert(!std::is_aggregate_v<TestRecursive0Member *>);
    static_assert(boost::pfr::tuple_size_v<TestRecursive0Member *> == 1);
    static_assert(MosaicPattern<TestRecursive0Member *>);

    static_assert(std::is_scalar_v<TestRecursive1Member *>);
    static_assert(!std::is_aggregate_v<TestRecursive1Member *>);
    static_assert(boost::pfr::tuple_size_v<TestRecursive1Member *> == 1);
    static_assert(MosaicPattern<TestRecursive1Member *>);

    static_assert(std::is_scalar_v<TestRecursive2Members *>);
    static_assert(!std::is_aggregate_v<TestRecursive2Members *>);
    static_assert(boost::pfr::tuple_size_v<TestRecursive2Members *> == 1);
    static_assert(MosaicPattern<TestRecursive2Members *>);
  }
}

} // namespace ARIA
