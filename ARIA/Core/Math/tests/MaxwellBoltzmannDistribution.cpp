#include "ARIA/MaxwellBoltzmannDistribution.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

consteval bool StaticExpectEq(float a, float b) {
  float delta = a - b;
  if (delta < 0)
    delta = -delta;

  return delta < 1e-7;
}

} // namespace

TEST(MaxwellBoltzmannDistribution, D1) {
  using MBD = MaxwellBoltzmannDistribution<1, 1.5>;

  using Order0 = Tec<UInt<0>>;
  using Order1 = Tec<UInt<1>>;
  using Order2 = Tec<UInt<2>>;
  using Order3 = Tec<UInt<3>>;
  using Order4 = Tec<UInt<4>>;

  using DomainP = Tec<Int<1>>;
  using DomainO = Tec<Int<0>>;
  using DomainN = Tec<Int<-1>>;

  Tec1r u{0.114_R};
  Vec1r uV = ToVec(u);
  constexpr Tec1r uC{C<0.114_R>{}};

  auto testRuntime = [](auto u) {
    Real o0dP = MBD::Moment<Order0, DomainP>(u);
    Real o1dP = MBD::Moment<Order1, DomainP>(u);
    Real o2dP = MBD::Moment<Order2, DomainP>(u);
    Real o3dP = MBD::Moment<Order3, DomainP>(u);
    Real o4dP = MBD::Moment<Order4, DomainP>(u);

    Real o0dO = MBD::Moment<Order0, DomainO>(u);
    Real o1dO = MBD::Moment<Order1, DomainO>(u);
    Real o2dO = MBD::Moment<Order2, DomainO>(u);
    Real o3dO = MBD::Moment<Order3, DomainO>(u);
    Real o4dO = MBD::Moment<Order4, DomainO>(u);

    Real o0dN = MBD::Moment<Order0, DomainN>(u);
    Real o1dN = MBD::Moment<Order1, DomainN>(u);
    Real o2dN = MBD::Moment<Order2, DomainN>(u);
    Real o3dN = MBD::Moment<Order3, DomainN>(u);
    Real o4dN = MBD::Moment<Order4, DomainN>(u);

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
  };

  testRuntime(u);
  testRuntime(uV);

  static_assert(StaticExpectEq(MBD::Moment<Order0, DomainO>(uC), 1));
  static_assert(StaticExpectEq(MBD::Moment<Order1, DomainO>(uC), 0.114));
  static_assert(StaticExpectEq(MBD::Moment<Order2, DomainO>(uC), 0.3463293333));
  static_assert(StaticExpectEq(MBD::Moment<Order3, DomainO>(uC), 0.1154815440));
  static_assert(StaticExpectEq(MBD::Moment<Order4, DomainO>(uC), 0.3594942293));
}

TEST(MaxwellBoltzmannDistribution, D2) {
  using MBD = MaxwellBoltzmannDistribution<2, 1.5>;

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
  using DomainPO = Tec<Int<1>, Int<0>>;
  using DomainOO = Tec<Int<0>, Int<0>>;
  using DomainNO = Tec<Int<-1>, Int<0>>;
  using DomainPN = Tec<Int<1>, Int<-1>>;

  Tec2r u{0.114_R, 0.514_R};
  Vec2r uV = ToVec(u);
  constexpr Tec2r uC{C<0.114_R>{}, C<0.514_R>{}};

  auto testRuntime = [](auto u) {
    Real o00dPP = MBD::Moment<Order00, DomainPP>(u);
    Real o10dPP = MBD::Moment<Order10, DomainPP>(u);
    Real o01dPP = MBD::Moment<Order01, DomainPP>(u);
    Real o20dPP = MBD::Moment<Order20, DomainPP>(u);
    Real o11dPP = MBD::Moment<Order11, DomainPP>(u);
    Real o02dPP = MBD::Moment<Order02, DomainPP>(u);
    Real o30dPP = MBD::Moment<Order30, DomainPP>(u);
    Real o21dPP = MBD::Moment<Order21, DomainPP>(u);
    Real o12dPP = MBD::Moment<Order12, DomainPP>(u);
    Real o03dPP = MBD::Moment<Order03, DomainPP>(u);
    Real o40dPP = MBD::Moment<Order40, DomainPP>(u);
    Real o31dPP = MBD::Moment<Order31, DomainPP>(u);
    Real o22dPP = MBD::Moment<Order22, DomainPP>(u);
    Real o13dPP = MBD::Moment<Order13, DomainPP>(u);
    Real o04dPP = MBD::Moment<Order04, DomainPP>(u);

    Real o00dPO = MBD::Moment<Order00, DomainPO>(u);
    Real o10dPO = MBD::Moment<Order10, DomainPO>(u);
    Real o01dPO = MBD::Moment<Order01, DomainPO>(u);
    Real o20dPO = MBD::Moment<Order20, DomainPO>(u);
    Real o11dPO = MBD::Moment<Order11, DomainPO>(u);
    Real o02dPO = MBD::Moment<Order02, DomainPO>(u);
    Real o30dPO = MBD::Moment<Order30, DomainPO>(u);
    Real o21dPO = MBD::Moment<Order21, DomainPO>(u);
    Real o12dPO = MBD::Moment<Order12, DomainPO>(u);
    Real o03dPO = MBD::Moment<Order03, DomainPO>(u);
    Real o40dPO = MBD::Moment<Order40, DomainPO>(u);
    Real o31dPO = MBD::Moment<Order31, DomainPO>(u);
    Real o22dPO = MBD::Moment<Order22, DomainPO>(u);
    Real o13dPO = MBD::Moment<Order13, DomainPO>(u);
    Real o04dPO = MBD::Moment<Order04, DomainPO>(u);

    Real o00dOO = MBD::Moment<Order00, DomainOO>(u);
    Real o10dOO = MBD::Moment<Order10, DomainOO>(u);
    Real o01dOO = MBD::Moment<Order01, DomainOO>(u);
    Real o20dOO = MBD::Moment<Order20, DomainOO>(u);
    Real o11dOO = MBD::Moment<Order11, DomainOO>(u);
    Real o02dOO = MBD::Moment<Order02, DomainOO>(u);
    Real o30dOO = MBD::Moment<Order30, DomainOO>(u);
    Real o21dOO = MBD::Moment<Order21, DomainOO>(u);
    Real o12dOO = MBD::Moment<Order12, DomainOO>(u);
    Real o03dOO = MBD::Moment<Order03, DomainOO>(u);
    Real o40dOO = MBD::Moment<Order40, DomainOO>(u);
    Real o31dOO = MBD::Moment<Order31, DomainOO>(u);
    Real o22dOO = MBD::Moment<Order22, DomainOO>(u);
    Real o13dOO = MBD::Moment<Order13, DomainOO>(u);
    Real o04dOO = MBD::Moment<Order04, DomainOO>(u);

    Real o00dNO = MBD::Moment<Order00, DomainNO>(u);
    Real o10dNO = MBD::Moment<Order10, DomainNO>(u);
    Real o01dNO = MBD::Moment<Order01, DomainNO>(u);
    Real o20dNO = MBD::Moment<Order20, DomainNO>(u);
    Real o11dNO = MBD::Moment<Order11, DomainNO>(u);
    Real o02dNO = MBD::Moment<Order02, DomainNO>(u);
    Real o30dNO = MBD::Moment<Order30, DomainNO>(u);
    Real o21dNO = MBD::Moment<Order21, DomainNO>(u);
    Real o12dNO = MBD::Moment<Order12, DomainNO>(u);
    Real o03dNO = MBD::Moment<Order03, DomainNO>(u);
    Real o40dNO = MBD::Moment<Order40, DomainNO>(u);
    Real o31dNO = MBD::Moment<Order31, DomainNO>(u);
    Real o22dNO = MBD::Moment<Order22, DomainNO>(u);
    Real o13dNO = MBD::Moment<Order13, DomainNO>(u);
    Real o04dNO = MBD::Moment<Order04, DomainNO>(u);

    Real o00dPN = MBD::Moment<Order00, DomainPN>(u);
    Real o10dPN = MBD::Moment<Order10, DomainPN>(u);
    Real o01dPN = MBD::Moment<Order01, DomainPN>(u);
    Real o20dPN = MBD::Moment<Order20, DomainPN>(u);
    Real o11dPN = MBD::Moment<Order11, DomainPN>(u);
    Real o02dPN = MBD::Moment<Order02, DomainPN>(u);
    Real o30dPN = MBD::Moment<Order30, DomainPN>(u);
    Real o21dPN = MBD::Moment<Order21, DomainPN>(u);
    Real o12dPN = MBD::Moment<Order12, DomainPN>(u);
    Real o03dPN = MBD::Moment<Order03, DomainPN>(u);
    Real o40dPN = MBD::Moment<Order40, DomainPN>(u);
    Real o31dPN = MBD::Moment<Order31, DomainPN>(u);
    Real o22dPN = MBD::Moment<Order22, DomainPN>(u);
    Real o13dPN = MBD::Moment<Order13, DomainPN>(u);
    Real o04dPN = MBD::Moment<Order04, DomainPN>(u);

    EXPECT_FLOAT_EQ(o00dPP, 0.4703254347);
    EXPECT_FLOAT_EQ(o10dPP, 0.2373368200);
    EXPECT_FLOAT_EQ(o01dPP, 0.3313594720);
    EXPECT_FLOAT_EQ(o20dPP, 0.1838315424);
    EXPECT_FLOAT_EQ(o11dPP, 0.1672114616);
    EXPECT_FLOAT_EQ(o02dPP, 0.3270939135);
    EXPECT_FLOAT_EQ(o30dPP, 0.1791813425);
    EXPECT_FLOAT_EQ(o21dPP, 0.1295152639);
    EXPECT_FLOAT_EQ(o12dPP, 0.1650589646);
    EXPECT_FLOAT_EQ(o03dPP, 0.3890325862);
    EXPECT_FLOAT_EQ(o40dPP, 0.2042582154);
    EXPECT_FLOAT_EQ(o31dPP, 0.1262390478);
    EXPECT_FLOAT_EQ(o22dPP, 0.1278480265);
    EXPECT_FLOAT_EQ(o13dPP, 0.1963146155);
    EXPECT_FLOAT_EQ(o04dPP, 0.5270566628);

    EXPECT_FLOAT_EQ(o00dPO, 0.5782637809);
    EXPECT_FLOAT_EQ(o10dPO, 0.2918049435);
    EXPECT_FLOAT_EQ(o01dPO, 0.2972275834);
    EXPECT_FLOAT_EQ(o20dPO, 0.2260203572);
    EXPECT_FLOAT_EQ(o11dPO, 0.1499877409);
    EXPECT_FLOAT_EQ(o02dPO, 0.3455295715);
    EXPECT_FLOAT_EQ(o30dPO, 0.2203029497);
    EXPECT_FLOAT_EQ(o21dPO, 0.1161744636);
    EXPECT_FLOAT_EQ(o12dPO, 0.1743620133);
    EXPECT_FLOAT_EQ(o03dPO, 0.3757539220);
    EXPECT_FLOAT_EQ(o40dPO, 0.2511348935);
    EXPECT_FLOAT_EQ(o31dPO, 0.1132357161);
    EXPECT_FLOAT_EQ(o22dPO, 0.1350537934);
    EXPECT_FLOAT_EQ(o13dPO, 0.1896139022);
    EXPECT_FLOAT_EQ(o04dPO, 0.5386670874);

    EXPECT_FLOAT_EQ(o00dOO, 1);
    EXPECT_FLOAT_EQ(o10dOO, 0.114);
    EXPECT_FLOAT_EQ(o01dOO, 0.514);
    EXPECT_FLOAT_EQ(o20dOO, 0.3463293333);
    EXPECT_FLOAT_EQ(o11dOO, 0.0585960000);
    EXPECT_FLOAT_EQ(o02dOO, 0.5975293333);
    EXPECT_FLOAT_EQ(o30dOO, 0.1154815440);
    EXPECT_FLOAT_EQ(o21dOO, 0.1780132773);
    EXPECT_FLOAT_EQ(o12dOO, 0.0681183440);
    EXPECT_FLOAT_EQ(o03dOO, 0.6497967440);
    EXPECT_FLOAT_EQ(o40dOO, 0.3594942293);
    EXPECT_FLOAT_EQ(o31dOO, 0.0593575136);
    EXPECT_FLOAT_EQ(o22dOO, 0.2069419357);
    EXPECT_FLOAT_EQ(o13dOO, 0.0740768288);
    EXPECT_FLOAT_EQ(o04dOO, 0.9315248597);

    EXPECT_FLOAT_EQ(o00dNO, 0.4217362191);
    EXPECT_FLOAT_EQ(o10dNO, -0.1778049435);
    EXPECT_FLOAT_EQ(o01dNO, 0.2167724166);
    EXPECT_FLOAT_EQ(o20dNO, 0.1203089761);
    EXPECT_FLOAT_EQ(o11dNO, -0.0913917409);
    EXPECT_FLOAT_EQ(o02dNO, 0.2519997618);
    EXPECT_FLOAT_EQ(o30dNO, -0.1048214057);
    EXPECT_FLOAT_EQ(o21dNO, 0.0618388137);
    EXPECT_FLOAT_EQ(o12dNO, -0.1062436693);
    EXPECT_FLOAT_EQ(o03dNO, 0.2740428220);
    EXPECT_FLOAT_EQ(o40dNO, 0.1083593359);
    EXPECT_FLOAT_EQ(o31dNO, -0.0538782025);
    EXPECT_FLOAT_EQ(o22dNO, 0.0718881423);
    EXPECT_FLOAT_EQ(o13dNO, -0.1155370733);
    EXPECT_FLOAT_EQ(o04dNO, 0.3928577723);

    EXPECT_FLOAT_EQ(o00dPN, 0.1079383463);
    EXPECT_FLOAT_EQ(o10dPN, 0.0544681235);
    EXPECT_FLOAT_EQ(o01dPN, -0.0341318886);
    EXPECT_FLOAT_EQ(o20dPN, 0.0421888148);
    EXPECT_FLOAT_EQ(o11dPN, -0.0172237206);
    EXPECT_FLOAT_EQ(o02dPN, 0.0184356580);
    EXPECT_FLOAT_EQ(o30dPN, 0.0411216072);
    EXPECT_FLOAT_EQ(o21dPN, -0.0133408003);
    EXPECT_FLOAT_EQ(o12dPN, 0.0093030488);
    EXPECT_FLOAT_EQ(o03dPN, -0.0132786642);
    EXPECT_FLOAT_EQ(o40dPN, 0.0468766781);
    EXPECT_FLOAT_EQ(o31dPN, -0.0130033317);
    EXPECT_FLOAT_EQ(o22dPN, 0.0072057669);
    EXPECT_FLOAT_EQ(o13dPN, -0.0067007134);
    EXPECT_FLOAT_EQ(o04dPN, 0.011610424655 - 0.00000000005);
  };

  testRuntime(u);
  testRuntime(uV);

  static_assert(StaticExpectEq(MBD::Moment<Order00, DomainOO>(uC), 1));
  static_assert(StaticExpectEq(MBD::Moment<Order10, DomainOO>(uC), 0.114));
  static_assert(StaticExpectEq(MBD::Moment<Order01, DomainOO>(uC), 0.514));
  static_assert(StaticExpectEq(MBD::Moment<Order20, DomainOO>(uC), 0.3463293333));
  static_assert(StaticExpectEq(MBD::Moment<Order11, DomainOO>(uC), 0.0585960000));
  static_assert(StaticExpectEq(MBD::Moment<Order02, DomainOO>(uC), 0.5975293333));
  static_assert(StaticExpectEq(MBD::Moment<Order30, DomainOO>(uC), 0.1154815440));
  static_assert(StaticExpectEq(MBD::Moment<Order21, DomainOO>(uC), 0.1780132773));
  static_assert(StaticExpectEq(MBD::Moment<Order12, DomainOO>(uC), 0.0681183440));
  static_assert(StaticExpectEq(MBD::Moment<Order03, DomainOO>(uC), 0.6497967440));
  static_assert(StaticExpectEq(MBD::Moment<Order40, DomainOO>(uC), 0.3594942293));
  static_assert(StaticExpectEq(MBD::Moment<Order31, DomainOO>(uC), 0.0593575136));
  static_assert(StaticExpectEq(MBD::Moment<Order22, DomainOO>(uC), 0.2069419357));
  static_assert(StaticExpectEq(MBD::Moment<Order13, DomainOO>(uC), 0.0740768288));
  static_assert(StaticExpectEq(MBD::Moment<Order04, DomainOO>(uC), 0.9315248597));
}

TEST(MaxwellBoltzmannDistribution, D3) {
  using MBD = MaxwellBoltzmannDistribution<3, 1.5>;

  using Order000 = Tec<UInt<0>, UInt<0>, UInt<0>>;
  using Order100 = Tec<UInt<1>, UInt<0>, UInt<0>>;
  using Order010 = Tec<UInt<0>, UInt<1>, UInt<0>>;
  using Order001 = Tec<UInt<0>, UInt<0>, UInt<1>>;
  using Order200 = Tec<UInt<2>, UInt<0>, UInt<0>>;
  using Order110 = Tec<UInt<1>, UInt<1>, UInt<0>>;
  using Order020 = Tec<UInt<0>, UInt<2>, UInt<0>>;
  using Order101 = Tec<UInt<1>, UInt<0>, UInt<1>>;
  using Order011 = Tec<UInt<0>, UInt<1>, UInt<1>>;
  using Order002 = Tec<UInt<0>, UInt<0>, UInt<2>>;
  using Order300 = Tec<UInt<3>, UInt<0>, UInt<0>>;
  using Order210 = Tec<UInt<2>, UInt<1>, UInt<0>>;
  using Order120 = Tec<UInt<1>, UInt<2>, UInt<0>>;
  using Order030 = Tec<UInt<0>, UInt<3>, UInt<0>>;
  using Order201 = Tec<UInt<2>, UInt<0>, UInt<1>>;
  using Order111 = Tec<UInt<1>, UInt<1>, UInt<1>>;
  using Order021 = Tec<UInt<0>, UInt<2>, UInt<1>>;
  using Order102 = Tec<UInt<1>, UInt<0>, UInt<2>>;
  using Order012 = Tec<UInt<0>, UInt<1>, UInt<2>>;
  using Order003 = Tec<UInt<0>, UInt<0>, UInt<3>>;

  using DomainPOO = Tec<Int<1>, Int<0>, Int<0>>;
  using DomainOOO = Tec<Int<0>, Int<0>, Int<0>>;
  using DomainNOO = Tec<Int<-1>, Int<0>, Int<0>>;

  Tec3r u{0.114_R, 0.514_R, 0.1919810_R};
  Vec3r uV = ToVec(u);
  constexpr Tec3r uC{C<0.114_R>{}, C<0.514_R>{}, C<0.1919810_R>{}};

  auto testRuntime = [](auto u) {
    Real o000dPOO = MBD::Moment<Order000, DomainPOO>(u);
    Real o100dPOO = MBD::Moment<Order100, DomainPOO>(u);
    Real o010dPOO = MBD::Moment<Order010, DomainPOO>(u);
    Real o001dPOO = MBD::Moment<Order001, DomainPOO>(u);
    Real o200dPOO = MBD::Moment<Order200, DomainPOO>(u);
    Real o110dPOO = MBD::Moment<Order110, DomainPOO>(u);
    Real o020dPOO = MBD::Moment<Order020, DomainPOO>(u);
    Real o101dPOO = MBD::Moment<Order101, DomainPOO>(u);
    Real o011dPOO = MBD::Moment<Order011, DomainPOO>(u);
    Real o002dPOO = MBD::Moment<Order002, DomainPOO>(u);
    Real o300dPOO = MBD::Moment<Order300, DomainPOO>(u);
    Real o210dPOO = MBD::Moment<Order210, DomainPOO>(u);
    Real o120dPOO = MBD::Moment<Order120, DomainPOO>(u);
    Real o030dPOO = MBD::Moment<Order030, DomainPOO>(u);
    Real o201dPOO = MBD::Moment<Order201, DomainPOO>(u);
    Real o111dPOO = MBD::Moment<Order111, DomainPOO>(u);
    Real o021dPOO = MBD::Moment<Order021, DomainPOO>(u);
    Real o102dPOO = MBD::Moment<Order102, DomainPOO>(u);
    Real o012dPOO = MBD::Moment<Order012, DomainPOO>(u);
    Real o003dPOO = MBD::Moment<Order003, DomainPOO>(u);

    Real o000dOOO = MBD::Moment<Order000, DomainOOO>(u);
    Real o100dOOO = MBD::Moment<Order100, DomainOOO>(u);
    Real o010dOOO = MBD::Moment<Order010, DomainOOO>(u);
    Real o001dOOO = MBD::Moment<Order001, DomainOOO>(u);
    Real o200dOOO = MBD::Moment<Order200, DomainOOO>(u);
    Real o110dOOO = MBD::Moment<Order110, DomainOOO>(u);
    Real o020dOOO = MBD::Moment<Order020, DomainOOO>(u);
    Real o101dOOO = MBD::Moment<Order101, DomainOOO>(u);
    Real o011dOOO = MBD::Moment<Order011, DomainOOO>(u);
    Real o002dOOO = MBD::Moment<Order002, DomainOOO>(u);
    Real o300dOOO = MBD::Moment<Order300, DomainOOO>(u);
    Real o210dOOO = MBD::Moment<Order210, DomainOOO>(u);
    Real o120dOOO = MBD::Moment<Order120, DomainOOO>(u);
    Real o030dOOO = MBD::Moment<Order030, DomainOOO>(u);
    Real o201dOOO = MBD::Moment<Order201, DomainOOO>(u);
    Real o111dOOO = MBD::Moment<Order111, DomainOOO>(u);
    Real o021dOOO = MBD::Moment<Order021, DomainOOO>(u);
    Real o102dOOO = MBD::Moment<Order102, DomainOOO>(u);
    Real o012dOOO = MBD::Moment<Order012, DomainOOO>(u);
    Real o003dOOO = MBD::Moment<Order003, DomainOOO>(u);

    Real o000dNOO = MBD::Moment<Order000, DomainNOO>(u);
    Real o100dNOO = MBD::Moment<Order100, DomainNOO>(u);
    Real o010dNOO = MBD::Moment<Order010, DomainNOO>(u);
    Real o001dNOO = MBD::Moment<Order001, DomainNOO>(u);
    Real o200dNOO = MBD::Moment<Order200, DomainNOO>(u);
    Real o110dNOO = MBD::Moment<Order110, DomainNOO>(u);
    Real o020dNOO = MBD::Moment<Order020, DomainNOO>(u);
    Real o101dNOO = MBD::Moment<Order101, DomainNOO>(u);
    Real o011dNOO = MBD::Moment<Order011, DomainNOO>(u);
    Real o002dNOO = MBD::Moment<Order002, DomainNOO>(u);
    Real o300dNOO = MBD::Moment<Order300, DomainNOO>(u);
    Real o210dNOO = MBD::Moment<Order210, DomainNOO>(u);
    Real o120dNOO = MBD::Moment<Order120, DomainNOO>(u);
    Real o030dNOO = MBD::Moment<Order030, DomainNOO>(u);
    Real o201dNOO = MBD::Moment<Order201, DomainNOO>(u);
    Real o111dNOO = MBD::Moment<Order111, DomainNOO>(u);
    Real o021dNOO = MBD::Moment<Order021, DomainNOO>(u);
    Real o102dNOO = MBD::Moment<Order102, DomainNOO>(u);
    Real o012dNOO = MBD::Moment<Order012, DomainNOO>(u);
    Real o003dNOO = MBD::Moment<Order003, DomainNOO>(u);

    EXPECT_FLOAT_EQ(o000dPOO, 0.5782637809);
    EXPECT_FLOAT_EQ(o100dPOO, 0.2918049435);
    EXPECT_FLOAT_EQ(o010dPOO, 0.2972275834);
    EXPECT_FLOAT_EQ(o001dPOO, 0.1110156589);
    EXPECT_FLOAT_EQ(o200dPOO, 0.2260203572);
    EXPECT_FLOAT_EQ(o110dPOO, 0.1499877409);
    EXPECT_FLOAT_EQ(o020dPOO, 0.3455295715);
    EXPECT_FLOAT_EQ(o101dPOO, 0.0560210049);
    EXPECT_FLOAT_EQ(o011dPOO, 0.0570620487);
    EXPECT_FLOAT_EQ(o002dPOO, 0.2140674909);
    EXPECT_FLOAT_EQ(o300dPOO, 0.2203029497);
    EXPECT_FLOAT_EQ(o210dPOO, 0.1161744636);
    EXPECT_FLOAT_EQ(o120dPOO, 0.1743620133);
    EXPECT_FLOAT_EQ(o030dPOO, 0.3757539220);
    EXPECT_FLOAT_EQ(o201dPOO, 0.0433916142);
    EXPECT_FLOAT_EQ(o111dPOO, 0.0287947965);
    EXPECT_FLOAT_EQ(o021dPOO, 0.0663351127);
    EXPECT_FLOAT_EQ(o102dPOO, 0.1080232830);
    EXPECT_FLOAT_EQ(o012dPOO, 0.1100306903);
    EXPECT_FLOAT_EQ(o003dPOO, 0.1151073302);

    EXPECT_FLOAT_EQ(o000dOOO, 1);
    EXPECT_FLOAT_EQ(o100dOOO, 0.114);
    EXPECT_FLOAT_EQ(o010dOOO, 0.514);
    EXPECT_FLOAT_EQ(o001dOOO, 0.1919810);
    EXPECT_FLOAT_EQ(o200dOOO, 0.3463293333);
    EXPECT_FLOAT_EQ(o110dOOO, 0.0585960000);
    EXPECT_FLOAT_EQ(o020dOOO, 0.5975293333);
    EXPECT_FLOAT_EQ(o101dOOO, 0.0218858340);
    EXPECT_FLOAT_EQ(o011dOOO, 0.0986782340);
    EXPECT_FLOAT_EQ(o002dOOO, 0.3701900377);
    EXPECT_FLOAT_EQ(o300dOOO, 0.1154815440);
    EXPECT_FLOAT_EQ(o210dOOO, 0.1780132773);
    EXPECT_FLOAT_EQ(o120dOOO, 0.0681183440);
    EXPECT_FLOAT_EQ(o030dOOO, 0.6497967440);
    EXPECT_FLOAT_EQ(o201dOOO, 0.0664886517);
    EXPECT_FLOAT_EQ(o111dOOO, 0.0112493187);
    EXPECT_FLOAT_EQ(o021dOOO, 0.1147142789);
    EXPECT_FLOAT_EQ(o102dOOO, 0.0422016643);
    EXPECT_FLOAT_EQ(o012dOOO, 0.1902776794);
    EXPECT_FLOAT_EQ(o003dOOO, 0.1990567870);

    EXPECT_FLOAT_EQ(o000dNOO, 0.4217362191);
    EXPECT_FLOAT_EQ(o100dNOO, -0.1778049435);
    EXPECT_FLOAT_EQ(o010dNOO, 0.2167724166);
    EXPECT_FLOAT_EQ(o001dNOO, 0.0809653411);
    EXPECT_FLOAT_EQ(o200dNOO, 0.1203089761);
    EXPECT_FLOAT_EQ(o110dNOO, -0.0913917409);
    EXPECT_FLOAT_EQ(o020dNOO, 0.2519997618);
    EXPECT_FLOAT_EQ(o101dNOO, -0.0341351709);
    EXPECT_FLOAT_EQ(o011dNOO, 0.0416161853);
    EXPECT_FLOAT_EQ(o002dNOO, 0.1561225468);
    EXPECT_FLOAT_EQ(o300dNOO, -0.1048214057);
    EXPECT_FLOAT_EQ(o210dNOO, 0.0618388137);
    EXPECT_FLOAT_EQ(o120dNOO, -0.1062436693);
    EXPECT_FLOAT_EQ(o030dNOO, 0.2740428220);
    EXPECT_FLOAT_EQ(o201dNOO, 0.0230970375);
    EXPECT_FLOAT_EQ(o111dNOO, -0.0175454778);
    EXPECT_FLOAT_EQ(o021dNOO, 0.0483791663);
    EXPECT_FLOAT_EQ(o102dNOO, -0.0658216187);
    EXPECT_FLOAT_EQ(o012dNOO, 0.0802469891);
    EXPECT_FLOAT_EQ(o003dNOO, 0.0839494567);
  };

  testRuntime(u);
  testRuntime(uV);

  static_assert(StaticExpectEq(MBD::Moment<Order000, DomainOOO>(uC), 1));
  static_assert(StaticExpectEq(MBD::Moment<Order100, DomainOOO>(uC), 0.114));
  static_assert(StaticExpectEq(MBD::Moment<Order010, DomainOOO>(uC), 0.514));
  static_assert(StaticExpectEq(MBD::Moment<Order001, DomainOOO>(uC), 0.1919810));
  static_assert(StaticExpectEq(MBD::Moment<Order200, DomainOOO>(uC), 0.3463293333));
  static_assert(StaticExpectEq(MBD::Moment<Order110, DomainOOO>(uC), 0.0585960000));
  static_assert(StaticExpectEq(MBD::Moment<Order020, DomainOOO>(uC), 0.5975293333));
  static_assert(StaticExpectEq(MBD::Moment<Order101, DomainOOO>(uC), 0.0218858340));
  static_assert(StaticExpectEq(MBD::Moment<Order011, DomainOOO>(uC), 0.0986782340));
  static_assert(StaticExpectEq(MBD::Moment<Order002, DomainOOO>(uC), 0.3701900377));
  static_assert(StaticExpectEq(MBD::Moment<Order300, DomainOOO>(uC), 0.1154815440));
  static_assert(StaticExpectEq(MBD::Moment<Order210, DomainOOO>(uC), 0.1780132773));
  static_assert(StaticExpectEq(MBD::Moment<Order120, DomainOOO>(uC), 0.0681183440));
  static_assert(StaticExpectEq(MBD::Moment<Order030, DomainOOO>(uC), 0.6497967440));
  static_assert(StaticExpectEq(MBD::Moment<Order201, DomainOOO>(uC), 0.0664886517));
  static_assert(StaticExpectEq(MBD::Moment<Order111, DomainOOO>(uC), 0.0112493187));
  static_assert(StaticExpectEq(MBD::Moment<Order021, DomainOOO>(uC), 0.1147142789));
  static_assert(StaticExpectEq(MBD::Moment<Order102, DomainOOO>(uC), 0.0422016643));
  static_assert(StaticExpectEq(MBD::Moment<Order012, DomainOOO>(uC), 0.1902776794));
  static_assert(StaticExpectEq(MBD::Moment<Order003, DomainOOO>(uC), 0.1990567870));
}

} // namespace ARIA
