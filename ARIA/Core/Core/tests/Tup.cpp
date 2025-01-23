#include "ARIA/Tup.h"
#include "ARIA/Let.h"

#include <gtest/gtest.h>

namespace ARIA {

using cute::_0;
using cute::_1;
using cute::_2;
using cute::_3;
using cute::_4;

TEST(Tup, Base) {
  // Arithmetic type.
  static_assert(std::is_same_v<tup::detail::arithmetic_domain_t<int>, int>);
  static_assert(std::is_same_v<tup::detail::arithmetic_domain_t<const int>, int>);
  static_assert(std::is_same_v<tup::detail::arithmetic_domain_t<const int &>, int>);
  static_assert(std::is_same_v<tup::detail::arithmetic_domain_t<C<1>>, int>);
  static_assert(std::is_same_v<tup::detail::arithmetic_domain_t<const C<1>>, int>);
  static_assert(std::is_same_v<tup::detail::arithmetic_domain_t<const C<1> &>, int>);
  static_assert(std::is_same_v<tup::detail::arithmetic_domain_t<std::string>, void>);

  // Make tup.
  {
    Tup v{1, 2.0F, Tup{3.0, std::string{"4"}}};
    let vSub = Tup{3.0, std::string{"4"}};
    EXPECT_EQ(get<0>(v), 1);
    EXPECT_EQ(get<1>(v), 2.0F);
    EXPECT_EQ(get<2>(v), vSub);
    EXPECT_EQ(get<0>(get<2>(v)), 3.0);
    EXPECT_EQ(get<1>(get<2>(v)), std::string{"4"});

    static_assert(!tup::detail::is_tec_v<decltype(v)>);
    static_assert(!tup::detail::is_tec_v<decltype(vSub)>);
  }

  // Make tec.
  {
    Tec v{1, 2.0F, C<3U>{}, C<4.0>{}};
    EXPECT_EQ(get<0>(v), 1);
    EXPECT_EQ(get<1>(v), 2.0F);
    static_assert(get<2>(v) == C<3U>{});
    static_assert(get<3>(v) == C<4.0>{});

    static_assert(tup::detail::is_tec_v<decltype(v)>);
  }

  {
    constexpr Tec1 v1{1};
    constexpr Tec2 v2{1, 2.0F};
    constexpr Tec3 v3{1, 2.0F, C<3U>{}};
    constexpr Tec4 v4{1, 2.0F, C<3U>{}, C<4.0>{}};

    static_assert(tup::detail::is_tec_v<std::decay_t<decltype(v1)>>);
    static_assert(tup::detail::is_tec_v<std::decay_t<decltype(v2)>>);
    static_assert(tup::detail::is_tec_v<std::decay_t<decltype(v3)>>);
    static_assert(tup::detail::is_tec_v<std::decay_t<decltype(v4)>>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(v1)>, 1>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(v2)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(v3)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(v4)>, 4>);
  }

  {
    constexpr Teci vi{1, C<2>{}, 3, C<4>{}, 5};
    constexpr Tecu vu{1U, C<2U>{}, 3U, C<4U>{}, 5U};
    constexpr Tecf vf{1.0F, C<2.0F>{}, 3.0F, C<4.0F>{}, 5.0F};
    constexpr Tecd vd{1.0, C<2.0>{}, 3.0, C<4.0>{}, 5.0};
    constexpr Tecr vr{1_R, C<2_R>{}, 3_R, C<4_R>{}, 5_R};

    static_assert(tup::detail::is_tec_v<std::decay_t<decltype(vi)>>);
    static_assert(tup::detail::is_tec_v<std::decay_t<decltype(vu)>>);
    static_assert(tup::detail::is_tec_v<std::decay_t<decltype(vf)>>);
    static_assert(tup::detail::is_tec_v<std::decay_t<decltype(vd)>>);
    static_assert(tup::detail::is_tec_v<std::decay_t<decltype(vr)>>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr)>, Real>);
  }

  {
    constexpr Tec1i vi_0{1};
    constexpr Tec1u vu_0{1U};
    constexpr Tec1f vf_0{1.0F};
    constexpr Tec1d vd_0{1.0};
    constexpr Tec1r vr_0{1_R};

    constexpr Tec1i vi_1{C<1>{}};
    constexpr Tec1u vu_1{C<1U>{}};
    constexpr Tec1f vf_1{C<1.0F>{}};
    constexpr Tec1d vd_1{C<1.0>{}};
    constexpr Tec1r vr_1{C<1_R>{}};

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_0)>, 1>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_0)>, 1>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_0)>, 1>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_0)>, 1>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_0)>, 1>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_1)>, 1>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_1)>, 1>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_1)>, 1>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_1)>, 1>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_1)>, 1>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_0)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_0)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_0)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_0)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_0)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_1)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_1)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_1)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_1)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_1)>, Real>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_0)>, int, 1>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_0)>, uint, 1>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_0)>, float, 1>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_0)>, double, 1>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_0)>, Real, 1>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_1)>, int, 1>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_1)>, uint, 1>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_1)>, float, 1>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_1)>, double, 1>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_1)>, Real, 1>);

    static_assert(!tup::detail::is_tec_r_v<std::decay_t<decltype(vi_0)>, 2>);
    static_assert(!tup::detail::is_tec_t_v<std::decay_t<decltype(vi_0)>, uint>);
    static_assert(!tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_0)>, int, 2>);
    static_assert(!tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_0)>, uint, 1>);
  }

  {
    constexpr Tec2i vi_00{1, 2};
    constexpr Tec2u vu_00{1U, 2U};
    constexpr Tec2f vf_00{1.0F, 2.0F};
    constexpr Tec2d vd_00{1.0, 2.0};
    constexpr Tec2r vr_00{1_R, 2_R};

    constexpr Tec2i vi_10{C<1>{}, 2};
    constexpr Tec2u vu_10{C<1U>{}, 2U};
    constexpr Tec2f vf_10{C<1.0F>{}, 2.0F};
    constexpr Tec2d vd_10{C<1.0>{}, 2.0};
    constexpr Tec2r vr_10{C<1_R>{}, 2_R};

    constexpr Tec2i vi_01{1, C<2>{}};
    constexpr Tec2u vu_01{1U, C<2U>{}};
    constexpr Tec2f vf_01{1.0F, C<2.0F>{}};
    constexpr Tec2d vd_01{1.0, C<2.0>{}};
    constexpr Tec2r vr_01{1_R, C<2_R>{}};

    constexpr Tec2i vi_11{C<1>{}, C<2>{}};
    constexpr Tec2u vu_11{C<1U>{}, C<2U>{}};
    constexpr Tec2f vf_11{C<1.0F>{}, C<2.0F>{}};
    constexpr Tec2d vd_11{C<1.0>{}, C<2.0>{}};
    constexpr Tec2r vr_11{C<1_R>{}, C<2_R>{}};

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_00)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_00)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_00)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_00)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_00)>, 2>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_10)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_10)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_10)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_10)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_10)>, 2>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_01)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_01)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_01)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_01)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_01)>, 2>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_11)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_11)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_11)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_11)>, 2>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_11)>, 2>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_00)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_00)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_00)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_00)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_00)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_10)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_10)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_10)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_10)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_10)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_01)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_01)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_01)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_01)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_01)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_11)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_11)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_11)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_11)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_11)>, Real>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_00)>, int, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_00)>, uint, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_00)>, float, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_00)>, double, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_00)>, Real, 2>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_10)>, int, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_10)>, uint, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_10)>, float, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_10)>, double, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_10)>, Real, 2>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_01)>, int, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_01)>, uint, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_01)>, float, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_01)>, double, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_01)>, Real, 2>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_11)>, int, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_11)>, uint, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_11)>, float, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_11)>, double, 2>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_11)>, Real, 2>);

    static_assert(!tup::detail::is_tec_r_v<std::decay_t<decltype(vi_00)>, 3>);
    static_assert(!tup::detail::is_tec_t_v<std::decay_t<decltype(vi_00)>, uint>);
    static_assert(!tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_00)>, int, 3>);
    static_assert(!tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_00)>, uint, 2>);
  }

  {
    constexpr Tec3i vi_000{1, 2, 3};
    constexpr Tec3u vu_000{1U, 2U, 3U};
    constexpr Tec3f vf_000{1.0F, 2.0F, 3.0F};
    constexpr Tec3d vd_000{1.0, 2.0, 3.0};
    constexpr Tec3r vr_000{1_R, 2_R, 3_R};

    constexpr Tec3i vi_100{C<1>{}, 2, 3};
    constexpr Tec3u vu_100{C<1U>{}, 2U, 3U};
    constexpr Tec3f vf_100{C<1.0F>{}, 2.0F, 3.0F};
    constexpr Tec3d vd_100{C<1.0>{}, 2.0, 3.0};
    constexpr Tec3r vr_100{C<1_R>{}, 2_R, 3_R};

    constexpr Tec3i vi_010{1, C<2>{}, 3};
    constexpr Tec3u vu_010{1U, C<2U>{}, 3U};
    constexpr Tec3f vf_010{1.0F, C<2.0F>{}, 3.0F};
    constexpr Tec3d vd_010{1.0, C<2.0>{}, 3.0};
    constexpr Tec3r vr_010{1_R, C<2_R>{}, 3_R};

    constexpr Tec3i vi_110{C<1>{}, C<2>{}, 3};
    constexpr Tec3u vu_110{C<1U>{}, C<2U>{}, 3U};
    constexpr Tec3f vf_110{C<1.0F>{}, C<2.0F>{}, 3.0F};
    constexpr Tec3d vd_110{C<1.0>{}, C<2.0>{}, 3.0};
    constexpr Tec3r vr_110{C<1_R>{}, C<2_R>{}, 3_R};

    constexpr Tec3i vi_001{1, 2, C<3>{}};
    constexpr Tec3u vu_001{1U, 2U, C<3U>{}};
    constexpr Tec3f vf_001{1.0F, 2.0F, C<3.0F>{}};
    constexpr Tec3d vd_001{1.0, 2.0, C<3.0>{}};
    constexpr Tec3r vr_001{1_R, 2_R, C<3_R>{}};

    constexpr Tec3i vi_101{C<1>{}, 2, C<3>{}};
    constexpr Tec3u vu_101{C<1U>{}, 2U, C<3U>{}};
    constexpr Tec3f vf_101{C<1.0F>{}, 2.0F, C<3.0F>{}};
    constexpr Tec3d vd_101{C<1.0>{}, 2.0, C<3.0>{}};
    constexpr Tec3r vr_101{C<1_R>{}, 2_R, C<3_R>{}};

    constexpr Tec3i vi_011{1, C<2>{}, C<3>{}};
    constexpr Tec3u vu_011{1U, C<2U>{}, C<3U>{}};
    constexpr Tec3f vf_011{1.0F, C<2.0F>{}, C<3.0F>{}};
    constexpr Tec3d vd_011{1.0, C<2.0>{}, C<3.0>{}};
    constexpr Tec3r vr_011{1_R, C<2_R>{}, C<3_R>{}};

    constexpr Tec3i vi_111{C<1>{}, C<2>{}, C<3>{}};
    constexpr Tec3u vu_111{C<1U>{}, C<2U>{}, C<3U>{}};
    constexpr Tec3f vf_111{C<1.0F>{}, C<2.0F>{}, C<3.0F>{}};
    constexpr Tec3d vd_111{C<1.0>{}, C<2.0>{}, C<3.0>{}};
    constexpr Tec3r vr_111{C<1_R>{}, C<2_R>{}, C<3_R>{}};

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_000)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_000)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_000)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_000)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_000)>, 3>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_100)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_100)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_100)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_100)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_100)>, 3>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_010)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_010)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_010)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_010)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_010)>, 3>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_110)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_110)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_110)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_110)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_110)>, 3>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_001)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_001)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_001)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_001)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_001)>, 3>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_101)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_101)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_101)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_101)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_101)>, 3>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_011)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_011)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_011)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_011)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_011)>, 3>);

    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vi_111)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vu_111)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vf_111)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vd_111)>, 3>);
    static_assert(tup::detail::is_tec_r_v<std::decay_t<decltype(vr_111)>, 3>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_000)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_000)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_000)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_000)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_000)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_100)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_100)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_100)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_100)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_100)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_010)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_010)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_010)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_010)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_010)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_110)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_110)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_110)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_110)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_110)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_001)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_001)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_001)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_001)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_001)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_101)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_101)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_101)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_101)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_101)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_011)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_011)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_011)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_011)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_011)>, Real>);

    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vi_111)>, int>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vu_111)>, uint>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vf_111)>, float>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vd_111)>, double>);
    static_assert(tup::detail::is_tec_t_v<std::decay_t<decltype(vr_111)>, Real>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_000)>, int, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_000)>, uint, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_000)>, float, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_000)>, double, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_000)>, Real, 3>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_100)>, int, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_100)>, uint, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_100)>, float, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_100)>, double, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_100)>, Real, 3>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_010)>, int, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_010)>, uint, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_010)>, float, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_010)>, double, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_010)>, Real, 3>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_110)>, int, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_110)>, uint, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_110)>, float, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_110)>, double, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_110)>, Real, 3>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_001)>, int, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_001)>, uint, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_001)>, float, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_001)>, double, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_001)>, Real, 3>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_101)>, int, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_101)>, uint, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_101)>, float, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_101)>, double, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_101)>, Real, 3>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_011)>, int, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_011)>, uint, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_011)>, float, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_011)>, double, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_011)>, Real, 3>);

    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_111)>, int, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vu_111)>, uint, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vf_111)>, float, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vd_111)>, double, 3>);
    static_assert(tup::detail::is_tec_tr_v<std::decay_t<decltype(vr_111)>, Real, 3>);

    static_assert(!tup::detail::is_tec_r_v<std::decay_t<decltype(vi_000)>, 4>);
    static_assert(!tup::detail::is_tec_t_v<std::decay_t<decltype(vi_000)>, uint>);
    static_assert(!tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_000)>, int, 4>);
    static_assert(!tup::detail::is_tec_tr_v<std::decay_t<decltype(vi_000)>, uint, 3>);
  }

  static_assert(rank(Tec{}) == 0);
  static_assert(rank(Tec{0}) == 1);
  static_assert(rank(Tec{_0{}}) == 1);
  static_assert(rank(Tec{0, 1}) == 2);
  static_assert(rank(Tec{_0{}, 1}) == 2);
  static_assert(rank(Tec{0, _1{}}) == 2);
  static_assert(rank(Tec{_0{}, _1{}}) == 2);

  static_assert(is_static_v<decltype(Tec{})>);
  static_assert(!is_static_v<decltype(Tec{0})>);
  static_assert(is_static_v<decltype(Tec{_0{}})>);
  static_assert(!is_static_v<decltype(Tec{0, 1})>);
  static_assert(!is_static_v<decltype(Tec{_0{}, 1})>);
  static_assert(!is_static_v<decltype(Tec{0, _1{}})>);
  static_assert(is_static_v<decltype(Tec{_0{}, _1{}})>);
}

} // namespace ARIA
