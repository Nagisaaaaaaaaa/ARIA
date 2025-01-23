#include "ARIA/BoltzmannDistribution.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

consteval bool StaticExpectEq(float a, float b) {
  float delta = a - b;
  if (delta < 0)
    delta = -delta;

  return delta < 1e-8;
}

} // namespace

TEST(BoltzmannDistribution, D1) {
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
  constexpr Tec1r uC{C<0.114_R>{}};

  Real o0dP = BD::Moment<Order0, DomainP>(u);
  Real o1dP = BD::Moment<Order1, DomainP>(u);
  Real o2dP = BD::Moment<Order2, DomainP>(u);
  Real o3dP = BD::Moment<Order3, DomainP>(u);
  Real o4dP = BD::Moment<Order4, DomainP>(u);

  Real o0dO = BD::Moment<Order0, DomainO>(u);
  Real o1dO = BD::Moment<Order1, DomainO>(u);
  Real o2dO = BD::Moment<Order2, DomainO>(u);
  Real o3dO = BD::Moment<Order3, DomainO>(u);
  Real o4dO = BD::Moment<Order4, DomainO>(u);

  Real o0dN = BD::Moment<Order0, DomainN>(u);
  Real o1dN = BD::Moment<Order1, DomainN>(u);
  Real o2dN = BD::Moment<Order2, DomainN>(u);
  Real o3dN = BD::Moment<Order3, DomainN>(u);
  Real o4dN = BD::Moment<Order4, DomainN>(u);

  EXPECT_FLOAT_EQ(o0dP, 0.5782637809);
  EXPECT_FLOAT_EQ(o1dP, 0.2918049435);
  EXPECT_FLOAT_EQ(o2dP, 0.2260203572);
  EXPECT_FLOAT_EQ(o3dP, 0.2203029497);
  EXPECT_FLOAT_EQ(o4dP, 0.2511348935);

  EXPECT_FLOAT_EQ(o0dO, 1);
  EXPECT_FLOAT_EQ(o1dO, 0.114);
  EXPECT_FLOAT_EQ(o2dO, 0.3463293333);
  EXPECT_FLOAT_EQ(o3dO, 0.1154815440);
  EXPECT_FLOAT_EQ(o4dO, 0.3594942293);

  EXPECT_FLOAT_EQ(o0dN, 0.4217362191);
  EXPECT_FLOAT_EQ(o1dN, -0.1778049435);
  EXPECT_FLOAT_EQ(o2dN, 0.1203089761);
  EXPECT_FLOAT_EQ(o3dN, -0.1048214057);
  EXPECT_FLOAT_EQ(o4dN, 0.1083593359);

  static_assert(StaticExpectEq(BD::Moment<Order0, DomainO>(uC), 1));
  static_assert(StaticExpectEq(BD::Moment<Order1, DomainO>(uC), 0.114));
  static_assert(StaticExpectEq(BD::Moment<Order2, DomainO>(uC), 0.3463293333));
  static_assert(StaticExpectEq(BD::Moment<Order3, DomainO>(uC), 0.1154815440));
  static_assert(StaticExpectEq(BD::Moment<Order4, DomainO>(uC), 0.3594942293));
}

TEST(BoltzmannDistribution, D2) {
  using BD = BoltzmannDistribution<2, 1.5>;

  using Order00 = Tec<UInt<0>, UInt<0>>;
  using Order10 = Tec<UInt<1>, UInt<0>>;
  using Order01 = Tec<UInt<0>, UInt<1>>;
  using Order20 = Tec<UInt<2>, UInt<0>>;
  using Order11 = Tec<UInt<1>, UInt<1>>;
  using Order02 = Tec<UInt<0>, UInt<2>>;
  using Order30 = Tec<UInt<3>, UInt<0>>;
  using Order21 = Tec<UInt<2>, UInt<1>>;
  using Order12 = Tec<UInt<1>, UInt<2>>;
  using Order03 = Tec<UInt<0>, UInt<3>>;
  using Order40 = Tec<UInt<4>, UInt<0>>;
  using Order31 = Tec<UInt<3>, UInt<1>>;
  using Order22 = Tec<UInt<2>, UInt<2>>;
  using Order13 = Tec<UInt<1>, UInt<3>>;
  using Order04 = Tec<UInt<0>, UInt<4>>;

  using DomainPP = Tec<Int<1>, Int<1>>;
  using DomainOP = Tec<Int<0>, Int<1>>;
  using DomainNP = Tec<Int<-1>, Int<1>>;
  using DomainPO = Tec<Int<1>, Int<0>>;
  using DomainOO = Tec<Int<0>, Int<0>>;
  using DomainNO = Tec<Int<-1>, Int<0>>;
  using DomainPN = Tec<Int<1>, Int<-1>>;
  using DomainON = Tec<Int<0>, Int<-1>>;
  using DomainNN = Tec<Int<-1>, Int<-1>>;

  Tec2r u{0.114_R, 0.514_R};
  constexpr Tec2r uC{C<0.114_R>{}, C<0.514_R>{}};
}

} // namespace ARIA
