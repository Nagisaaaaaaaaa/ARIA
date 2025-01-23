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

  Real o00dPP = BD::Moment<Order00, DomainPP>(u);
  Real o10dPP = BD::Moment<Order10, DomainPP>(u);
  Real o01dPP = BD::Moment<Order01, DomainPP>(u);
  Real o20dPP = BD::Moment<Order20, DomainPP>(u);
  Real o11dPP = BD::Moment<Order11, DomainPP>(u);
  Real o02dPP = BD::Moment<Order02, DomainPP>(u);
  Real o30dPP = BD::Moment<Order30, DomainPP>(u);
  Real o21dPP = BD::Moment<Order21, DomainPP>(u);
  Real o12dPP = BD::Moment<Order12, DomainPP>(u);
  Real o03dPP = BD::Moment<Order03, DomainPP>(u);
  Real o40dPP = BD::Moment<Order40, DomainPP>(u);
  Real o31dPP = BD::Moment<Order31, DomainPP>(u);
  Real o22dPP = BD::Moment<Order22, DomainPP>(u);
  Real o13dPP = BD::Moment<Order13, DomainPP>(u);
  Real o04dPP = BD::Moment<Order04, DomainPP>(u);

  Real o00dOP = BD::Moment<Order00, DomainOP>(u);
  Real o10dOP = BD::Moment<Order10, DomainOP>(u);
  Real o01dOP = BD::Moment<Order01, DomainOP>(u);
  Real o20dOP = BD::Moment<Order20, DomainOP>(u);
  Real o11dOP = BD::Moment<Order11, DomainOP>(u);
  Real o02dOP = BD::Moment<Order02, DomainOP>(u);
  Real o30dOP = BD::Moment<Order30, DomainOP>(u);
  Real o21dOP = BD::Moment<Order21, DomainOP>(u);
  Real o12dOP = BD::Moment<Order12, DomainOP>(u);
  Real o03dOP = BD::Moment<Order03, DomainOP>(u);
  Real o40dOP = BD::Moment<Order40, DomainOP>(u);
  Real o31dOP = BD::Moment<Order31, DomainOP>(u);
  Real o22dOP = BD::Moment<Order22, DomainOP>(u);
  Real o13dOP = BD::Moment<Order13, DomainOP>(u);
  Real o04dOP = BD::Moment<Order04, DomainOP>(u);

  Real o00dNP = BD::Moment<Order00, DomainNP>(u);
  Real o10dNP = BD::Moment<Order10, DomainNP>(u);
  Real o01dNP = BD::Moment<Order01, DomainNP>(u);
  Real o20dNP = BD::Moment<Order20, DomainNP>(u);
  Real o11dNP = BD::Moment<Order11, DomainNP>(u);
  Real o02dNP = BD::Moment<Order02, DomainNP>(u);
  Real o30dNP = BD::Moment<Order30, DomainNP>(u);
  Real o21dNP = BD::Moment<Order21, DomainNP>(u);
  Real o12dNP = BD::Moment<Order12, DomainNP>(u);
  Real o03dNP = BD::Moment<Order03, DomainNP>(u);
  Real o40dNP = BD::Moment<Order40, DomainNP>(u);
  Real o31dNP = BD::Moment<Order31, DomainNP>(u);
  Real o22dNP = BD::Moment<Order22, DomainNP>(u);
  Real o13dNP = BD::Moment<Order13, DomainNP>(u);
  Real o04dNP = BD::Moment<Order04, DomainNP>(u);

  Real o00dPO = BD::Moment<Order00, DomainPO>(u);
  Real o10dPO = BD::Moment<Order10, DomainPO>(u);
  Real o01dPO = BD::Moment<Order01, DomainPO>(u);
  Real o20dPO = BD::Moment<Order20, DomainPO>(u);
  Real o11dPO = BD::Moment<Order11, DomainPO>(u);
  Real o02dPO = BD::Moment<Order02, DomainPO>(u);
  Real o30dPO = BD::Moment<Order30, DomainPO>(u);
  Real o21dPO = BD::Moment<Order21, DomainPO>(u);
  Real o12dPO = BD::Moment<Order12, DomainPO>(u);
  Real o03dPO = BD::Moment<Order03, DomainPO>(u);
  Real o40dPO = BD::Moment<Order40, DomainPO>(u);
  Real o31dPO = BD::Moment<Order31, DomainPO>(u);
  Real o22dPO = BD::Moment<Order22, DomainPO>(u);
  Real o13dPO = BD::Moment<Order13, DomainPO>(u);
  Real o04dPO = BD::Moment<Order04, DomainPO>(u);

  Real o00dOO = BD::Moment<Order00, DomainOO>(u);
  Real o10dOO = BD::Moment<Order10, DomainOO>(u);
  Real o01dOO = BD::Moment<Order01, DomainOO>(u);
  Real o20dOO = BD::Moment<Order20, DomainOO>(u);
  Real o11dOO = BD::Moment<Order11, DomainOO>(u);
  Real o02dOO = BD::Moment<Order02, DomainOO>(u);
  Real o30dOO = BD::Moment<Order30, DomainOO>(u);
  Real o21dOO = BD::Moment<Order21, DomainOO>(u);
  Real o12dOO = BD::Moment<Order12, DomainOO>(u);
  Real o03dOO = BD::Moment<Order03, DomainOO>(u);
  Real o40dOO = BD::Moment<Order40, DomainOO>(u);
  Real o31dOO = BD::Moment<Order31, DomainOO>(u);
  Real o22dOO = BD::Moment<Order22, DomainOO>(u);
  Real o13dOO = BD::Moment<Order13, DomainOO>(u);
  Real o04dOO = BD::Moment<Order04, DomainOO>(u);

  Real o00dNO = BD::Moment<Order00, DomainNO>(u);
  Real o10dNO = BD::Moment<Order10, DomainNO>(u);
  Real o01dNO = BD::Moment<Order01, DomainNO>(u);
  Real o20dNO = BD::Moment<Order20, DomainNO>(u);
  Real o11dNO = BD::Moment<Order11, DomainNO>(u);
  Real o02dNO = BD::Moment<Order02, DomainNO>(u);
  Real o30dNO = BD::Moment<Order30, DomainNO>(u);
  Real o21dNO = BD::Moment<Order21, DomainNO>(u);
  Real o12dNO = BD::Moment<Order12, DomainNO>(u);
  Real o03dNO = BD::Moment<Order03, DomainNO>(u);
  Real o40dNO = BD::Moment<Order40, DomainNO>(u);
  Real o31dNO = BD::Moment<Order31, DomainNO>(u);
  Real o22dNO = BD::Moment<Order22, DomainNO>(u);
  Real o13dNO = BD::Moment<Order13, DomainNO>(u);
  Real o04dNO = BD::Moment<Order04, DomainNO>(u);

  Real o00dPN = BD::Moment<Order00, DomainPN>(u);
  Real o10dPN = BD::Moment<Order10, DomainPN>(u);
  Real o01dPN = BD::Moment<Order01, DomainPN>(u);
  Real o20dPN = BD::Moment<Order20, DomainPN>(u);
  Real o11dPN = BD::Moment<Order11, DomainPN>(u);
  Real o02dPN = BD::Moment<Order02, DomainPN>(u);
  Real o30dPN = BD::Moment<Order30, DomainPN>(u);
  Real o21dPN = BD::Moment<Order21, DomainPN>(u);
  Real o12dPN = BD::Moment<Order12, DomainPN>(u);
  Real o03dPN = BD::Moment<Order03, DomainPN>(u);
  Real o40dPN = BD::Moment<Order40, DomainPN>(u);
  Real o31dPN = BD::Moment<Order31, DomainPN>(u);
  Real o22dPN = BD::Moment<Order22, DomainPN>(u);
  Real o13dPN = BD::Moment<Order13, DomainPN>(u);
  Real o04dPN = BD::Moment<Order04, DomainPN>(u);

  Real o00dON = BD::Moment<Order00, DomainON>(u);
  Real o10dON = BD::Moment<Order10, DomainON>(u);
  Real o01dON = BD::Moment<Order01, DomainON>(u);
  Real o20dON = BD::Moment<Order20, DomainON>(u);
  Real o11dON = BD::Moment<Order11, DomainON>(u);
  Real o02dON = BD::Moment<Order02, DomainON>(u);
  Real o30dON = BD::Moment<Order30, DomainON>(u);
  Real o21dON = BD::Moment<Order21, DomainON>(u);
  Real o12dON = BD::Moment<Order12, DomainON>(u);
  Real o03dON = BD::Moment<Order03, DomainON>(u);
  Real o40dON = BD::Moment<Order40, DomainON>(u);
  Real o31dON = BD::Moment<Order31, DomainON>(u);
  Real o22dON = BD::Moment<Order22, DomainON>(u);
  Real o13dON = BD::Moment<Order13, DomainON>(u);
  Real o04dON = BD::Moment<Order04, DomainON>(u);

  Real o00dNN = BD::Moment<Order00, DomainNN>(u);
  Real o10dNN = BD::Moment<Order10, DomainNN>(u);
  Real o01dNN = BD::Moment<Order01, DomainNN>(u);
  Real o20dNN = BD::Moment<Order20, DomainNN>(u);
  Real o11dNN = BD::Moment<Order11, DomainNN>(u);
  Real o02dNN = BD::Moment<Order02, DomainNN>(u);
  Real o30dNN = BD::Moment<Order30, DomainNN>(u);
  Real o21dNN = BD::Moment<Order21, DomainNN>(u);
  Real o12dNN = BD::Moment<Order12, DomainNN>(u);
  Real o03dNN = BD::Moment<Order03, DomainNN>(u);
  Real o40dNN = BD::Moment<Order40, DomainNN>(u);
  Real o31dNN = BD::Moment<Order31, DomainNN>(u);
  Real o22dNN = BD::Moment<Order22, DomainNN>(u);
  Real o13dNN = BD::Moment<Order13, DomainNN>(u);
  Real o04dNN = BD::Moment<Order04, DomainNN>(u);
}

} // namespace ARIA
