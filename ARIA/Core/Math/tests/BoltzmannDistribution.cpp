#include "ARIA/BoltzmannDistribution.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(BoltzmannDistribution, D1Runtime) {
  using BD = BoltzmannDistribution<1, 1.5>;

  using Order0 = Tec<UInt<0>>;
  using Order1 = Tec<UInt<1>>;
  using Order2 = Tec<UInt<2>>;
  using Order3 = Tec<UInt<3>>;
  using Order4 = Tec<UInt<4>>;

  using DomainP = Tec<Int<1>>;
  using DomainO = Tec<Int<0>>;
  using DomainN = Tec<Int<-1>>;

  Tec1r u{0.114_R};

  Real o0dO = BD::Moment<Order0, DomainO>(u);
  Real o1dO = BD::Moment<Order1, DomainO>(u);
  Real o2dO = BD::Moment<Order2, DomainO>(u);
  Real o3dO = BD::Moment<Order3, DomainO>(u);
  Real o4dO = BD::Moment<Order4, DomainO>(u);

  EXPECT_FLOAT_EQ(o0dO, 1);
  EXPECT_FLOAT_EQ(o1dO, 0.114);
  EXPECT_FLOAT_EQ(o2dO, 0.3463293333);
  EXPECT_FLOAT_EQ(o3dO, 0.1154815440);
  EXPECT_FLOAT_EQ(o4dO, 0.3594942293);
}

} // namespace ARIA
