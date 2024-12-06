#include "ARIA/Mosaic.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct StructWith0Member {};

//! Structures with 1 member is considered unnecessary.
struct StructWith1Member {
  int v = 5;
};

struct StructWith2Members {
  int v0 = 5;
  double v1 = 6;
};

struct StructWithPrivateMembers {
private:
  int v_ = 5;
};

} // namespace

TEST(Mosaic, Base) {
  static_assert(std::is_scalar_v<int>);
  static_assert(!std::is_aggregate_v<int>);
  static_assert(boost::pfr::tuple_size_v<int> == 1);
  static_assert(is_mosaic_v<int>);

  static_assert(!std::is_scalar_v<StructWith0Member>);
  static_assert(std::is_aggregate_v<StructWith0Member>);
  static_assert(boost::pfr::tuple_size_v<StructWith0Member> == 0);
  static_assert(!is_mosaic_v<StructWith0Member>);

  static_assert(!std::is_scalar_v<StructWith1Member>);
  static_assert(std::is_aggregate_v<StructWith1Member>);
  // static_assert(boost::pfr::tuple_size_v<StructWith1Member> == 1);
  // static_assert(!is_mosaic_v<StructWith1Member>);

  static_assert(!std::is_scalar_v<StructWith2Members>);
  static_assert(std::is_aggregate_v<StructWith2Members>);
  static_assert(boost::pfr::tuple_size_v<StructWith2Members> == 2);
  static_assert(is_mosaic_v<StructWith2Members>);

  static_assert(!std::is_scalar_v<StructWithPrivateMembers>);
  static_assert(!std::is_aggregate_v<StructWithPrivateMembers>);
  // static_assert(boost::pfr::tuple_size_v<StructWithPrivateMembers> == 0);
  // static_assert(!is_mosaic_v<StructWithPrivateMembers>);

  static_assert(std::is_scalar_v<double *>);
  static_assert(!std::is_aggregate_v<double *>);
  static_assert(boost::pfr::tuple_size_v<double *> == 1);
  static_assert(is_mosaic_v<double *>);

  static_assert(std::is_scalar_v<StructWith0Member *>);
  static_assert(!std::is_aggregate_v<StructWith0Member *>);
  static_assert(boost::pfr::tuple_size_v<StructWith0Member *> == 1);
  static_assert(is_mosaic_v<StructWith0Member *>);

  static_assert(std::is_scalar_v<StructWith1Member *>);
  static_assert(!std::is_aggregate_v<StructWith1Member *>);
  static_assert(boost::pfr::tuple_size_v<StructWith1Member *> == 1);
  static_assert(is_mosaic_v<StructWith1Member *>);

  static_assert(std::is_scalar_v<StructWith2Members *>);
  static_assert(!std::is_aggregate_v<StructWith2Members *>);
  static_assert(boost::pfr::tuple_size_v<StructWith2Members *> == 1);
  static_assert(is_mosaic_v<StructWith2Members *>);

  static_assert(std::is_scalar_v<StructWithPrivateMembers *>);
  static_assert(!std::is_aggregate_v<StructWithPrivateMembers *>);
  static_assert(boost::pfr::tuple_size_v<StructWithPrivateMembers *> == 1);
  static_assert(is_mosaic_v<StructWithPrivateMembers *>);
}

} // namespace ARIA
