#include "ARIA/MortonCode.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

template <typename I>
void TestMortonEncode1D() {
  using Code = MortonCode<1>;
  using V = Vec1<I>;

  EXPECT_EQ(Code::Encode(V{0}), 0);
  EXPECT_EQ(Code::Encode(V{1}), 1);
  EXPECT_EQ(Code::Encode(V{2}), 2);
  EXPECT_EQ(Code::Encode(V{3}), 3);

  EXPECT_EQ(Code::Decode(I{0}), V{0});
  EXPECT_EQ(Code::Decode(I{1}), V{1});
  EXPECT_EQ(Code::Decode(I{2}), V{2});
  EXPECT_EQ(Code::Decode(I{3}), V{3});

  EXPECT_EQ(Code::Encode(V{4}), 4);
  EXPECT_EQ(Code::Encode(V{5}), 5);
  EXPECT_EQ(Code::Encode(V{6}), 6);
  EXPECT_EQ(Code::Encode(V{7}), 7);

  EXPECT_EQ(Code::Decode(I{4}), V{4});
  EXPECT_EQ(Code::Decode(I{5}), V{5});
  EXPECT_EQ(Code::Decode(I{6}), V{6});
  EXPECT_EQ(Code::Decode(I{7}), V{7});
}

template <typename I>
void TestMortonEncode2D() {
  using Code = MortonCode<2>;
  using V = Vec2<I>;

  EXPECT_EQ(Code::Encode(V{0, 0}), 0);
  EXPECT_EQ(Code::Encode(V{1, 0}), 1);
  EXPECT_EQ(Code::Encode(V{0, 1}), 2);
  EXPECT_EQ(Code::Encode(V{1, 1}), 3);

  EXPECT_EQ(Code::Decode(I{0}), (V{0, 0}));
  EXPECT_EQ(Code::Decode(I{1}), (V{1, 0}));
  EXPECT_EQ(Code::Decode(I{2}), (V{0, 1}));
  EXPECT_EQ(Code::Decode(I{3}), (V{1, 1}));

  EXPECT_EQ(Code::Encode(V{2, 0}), 4);
  EXPECT_EQ(Code::Encode(V{3, 0}), 5);
  EXPECT_EQ(Code::Encode(V{2, 1}), 6);
  EXPECT_EQ(Code::Encode(V{3, 1}), 7);

  EXPECT_EQ(Code::Decode(I{4}), (V{2, 0}));
  EXPECT_EQ(Code::Decode(I{5}), (V{3, 0}));
  EXPECT_EQ(Code::Decode(I{6}), (V{2, 1}));
  EXPECT_EQ(Code::Decode(I{7}), (V{3, 1}));

  EXPECT_EQ(Code::Encode(V{0, 2}), 8);
  EXPECT_EQ(Code::Encode(V{1, 2}), 9);
  EXPECT_EQ(Code::Encode(V{0, 3}), 10);
  EXPECT_EQ(Code::Encode(V{1, 3}), 11);

  EXPECT_EQ(Code::Decode(I{8}), (V{0, 2}));
  EXPECT_EQ(Code::Decode(I{9}), (V{1, 2}));
  EXPECT_EQ(Code::Decode(I{10}), (V{0, 3}));
  EXPECT_EQ(Code::Decode(I{11}), (V{1, 3}));

  EXPECT_EQ(Code::Encode(V{2, 2}), 12);
  EXPECT_EQ(Code::Encode(V{3, 2}), 13);
  EXPECT_EQ(Code::Encode(V{2, 3}), 14);
  EXPECT_EQ(Code::Encode(V{3, 3}), 15);

  EXPECT_EQ(Code::Decode(I{12}), (V{2, 2}));
  EXPECT_EQ(Code::Decode(I{13}), (V{3, 2}));
  EXPECT_EQ(Code::Decode(I{14}), (V{2, 3}));
  EXPECT_EQ(Code::Decode(I{15}), (V{3, 3}));

  EXPECT_EQ(Code::Encode(V{4, 0}), 16);
  EXPECT_EQ(Code::Encode(V{5, 0}), 17);
  EXPECT_EQ(Code::Encode(V{4, 1}), 18);
  EXPECT_EQ(Code::Encode(V{5, 1}), 19);

  EXPECT_EQ(Code::Decode(I{16}), (V{4, 0}));
  EXPECT_EQ(Code::Decode(I{17}), (V{5, 0}));
  EXPECT_EQ(Code::Decode(I{18}), (V{4, 1}));
  EXPECT_EQ(Code::Decode(I{19}), (V{5, 1}));

  EXPECT_EQ(Code::Encode(V{6, 0}), 20);
  EXPECT_EQ(Code::Encode(V{7, 0}), 21);
  EXPECT_EQ(Code::Encode(V{6, 1}), 22);
  EXPECT_EQ(Code::Encode(V{7, 1}), 23);

  EXPECT_EQ(Code::Decode(I{20}), (V{6, 0}));
  EXPECT_EQ(Code::Decode(I{21}), (V{7, 0}));
  EXPECT_EQ(Code::Decode(I{22}), (V{6, 1}));
  EXPECT_EQ(Code::Decode(I{23}), (V{7, 1}));

  if constexpr (std::is_same_v<I, int>) {
    EXPECT_EQ(Code::Encode(V{0xFFFF, 0xFFFF}), -1);
    EXPECT_EQ(Code::Decode(I{-1}), (V{0xFFFF, 0xFFFF}));
  } else {
    EXPECT_EQ(Code::Encode(V{0xFFFF, 0xFFFF}), 0xFFFFFFFF);
    EXPECT_EQ(Code::Decode(I{0xFFFFFFFF}), (V{0xFFFF, 0xFFFF}));
  }

  if constexpr (sizeof(I) == 8) {
    if constexpr (std::is_same_v<I, int64>) {
      EXPECT_EQ(Code::Encode(V{0xFFFFFFFF, 0xFFFFFFFF}), -1);
      EXPECT_EQ(Code::Decode(I{-1}), (V{0xFFFFFFFF, 0xFFFFFFFF}));
    } else {
      EXPECT_EQ(Code::Encode(V{0xFFFFFFFF, 0xFFFFFFFF}), 0xFFFFFFFFFFFFFFFFLLU);
      EXPECT_EQ(Code::Decode(I{0xFFFFFFFFFFFFFFFFLLU}), (V{0xFFFFFFFF, 0xFFFFFFFF}));
    }
  }
}

template <typename I>
void TestMortonEncode3D() {
  using Code = MortonCode<3>;
  using V = Vec3<I>;

  EXPECT_EQ(Code::Encode(V{0, 0, 0}), 0);
  EXPECT_EQ(Code::Encode(V{1, 0, 0}), 1);
  EXPECT_EQ(Code::Encode(V{0, 1, 0}), 2);
  EXPECT_EQ(Code::Encode(V{1, 1, 0}), 3);
  EXPECT_EQ(Code::Encode(V{0, 0, 1}), 4);
  EXPECT_EQ(Code::Encode(V{1, 0, 1}), 5);
  EXPECT_EQ(Code::Encode(V{0, 1, 1}), 6);
  EXPECT_EQ(Code::Encode(V{1, 1, 1}), 7);

  EXPECT_EQ(Code::Decode(I{0}), (V{0, 0, 0}));
  EXPECT_EQ(Code::Decode(I{1}), (V{1, 0, 0}));
  EXPECT_EQ(Code::Decode(I{2}), (V{0, 1, 0}));
  EXPECT_EQ(Code::Decode(I{3}), (V{1, 1, 0}));
  EXPECT_EQ(Code::Decode(I{4}), (V{0, 0, 1}));
  EXPECT_EQ(Code::Decode(I{5}), (V{1, 0, 1}));
  EXPECT_EQ(Code::Decode(I{6}), (V{0, 1, 1}));
  EXPECT_EQ(Code::Decode(I{7}), (V{1, 1, 1}));

  EXPECT_EQ(Code::Encode(V{2, 0, 0}), 8);
  EXPECT_EQ(Code::Encode(V{3, 0, 0}), 9);
  EXPECT_EQ(Code::Encode(V{2, 1, 0}), 10);
  EXPECT_EQ(Code::Encode(V{3, 1, 0}), 11);
  EXPECT_EQ(Code::Encode(V{2, 0, 1}), 12);
  EXPECT_EQ(Code::Encode(V{3, 0, 1}), 13);
  EXPECT_EQ(Code::Encode(V{2, 1, 1}), 14);
  EXPECT_EQ(Code::Encode(V{3, 1, 1}), 15);

  EXPECT_EQ(Code::Decode(I{8}), (V{2, 0, 0}));
  EXPECT_EQ(Code::Decode(I{9}), (V{3, 0, 0}));
  EXPECT_EQ(Code::Decode(I{10}), (V{2, 1, 0}));
  EXPECT_EQ(Code::Decode(I{11}), (V{3, 1, 0}));
  EXPECT_EQ(Code::Decode(I{12}), (V{2, 0, 1}));
  EXPECT_EQ(Code::Decode(I{13}), (V{3, 0, 1}));
  EXPECT_EQ(Code::Decode(I{14}), (V{2, 1, 1}));
  EXPECT_EQ(Code::Decode(I{15}), (V{3, 1, 1}));

  EXPECT_EQ(Code::Encode(V{0, 2, 0}), 16);
  EXPECT_EQ(Code::Encode(V{1, 2, 0}), 17);
  EXPECT_EQ(Code::Encode(V{0, 3, 0}), 18);
  EXPECT_EQ(Code::Encode(V{1, 3, 0}), 19);
  EXPECT_EQ(Code::Encode(V{0, 2, 1}), 20);
  EXPECT_EQ(Code::Encode(V{1, 2, 1}), 21);
  EXPECT_EQ(Code::Encode(V{0, 3, 1}), 22);
  EXPECT_EQ(Code::Encode(V{1, 3, 1}), 23);

  EXPECT_EQ(Code::Decode(I{16}), (V{0, 2, 0}));
  EXPECT_EQ(Code::Decode(I{17}), (V{1, 2, 0}));
  EXPECT_EQ(Code::Decode(I{18}), (V{0, 3, 0}));
  EXPECT_EQ(Code::Decode(I{19}), (V{1, 3, 0}));
  EXPECT_EQ(Code::Decode(I{20}), (V{0, 2, 1}));
  EXPECT_EQ(Code::Decode(I{21}), (V{1, 2, 1}));
  EXPECT_EQ(Code::Decode(I{22}), (V{0, 3, 1}));
  EXPECT_EQ(Code::Decode(I{23}), (V{1, 3, 1}));

  EXPECT_EQ(Code::Encode(V{2, 2, 0}), 24);
  EXPECT_EQ(Code::Encode(V{3, 2, 0}), 25);
  EXPECT_EQ(Code::Encode(V{2, 3, 0}), 26);
  EXPECT_EQ(Code::Encode(V{3, 3, 0}), 27);
  EXPECT_EQ(Code::Encode(V{2, 2, 1}), 28);
  EXPECT_EQ(Code::Encode(V{3, 2, 1}), 29);
  EXPECT_EQ(Code::Encode(V{2, 3, 1}), 30);
  EXPECT_EQ(Code::Encode(V{3, 3, 1}), 31);

  EXPECT_EQ(Code::Decode(I{24}), (V{2, 2, 0}));
  EXPECT_EQ(Code::Decode(I{25}), (V{3, 2, 0}));
  EXPECT_EQ(Code::Decode(I{26}), (V{2, 3, 0}));
  EXPECT_EQ(Code::Decode(I{27}), (V{3, 3, 0}));
  EXPECT_EQ(Code::Decode(I{28}), (V{2, 2, 1}));
  EXPECT_EQ(Code::Decode(I{29}), (V{3, 2, 1}));
  EXPECT_EQ(Code::Decode(I{30}), (V{2, 3, 1}));
  EXPECT_EQ(Code::Decode(I{31}), (V{3, 3, 1}));

  //
  //
  //
  EXPECT_EQ(Code::Encode(V{0, 0, 2}), 32);
  EXPECT_EQ(Code::Encode(V{1, 0, 2}), 33);
  EXPECT_EQ(Code::Encode(V{0, 1, 2}), 34);
  EXPECT_EQ(Code::Encode(V{1, 1, 2}), 35);
  EXPECT_EQ(Code::Encode(V{0, 0, 3}), 36);
  EXPECT_EQ(Code::Encode(V{1, 0, 3}), 37);
  EXPECT_EQ(Code::Encode(V{0, 1, 3}), 38);
  EXPECT_EQ(Code::Encode(V{1, 1, 3}), 39);

  EXPECT_EQ(Code::Decode(I{32}), (V{0, 0, 2}));
  EXPECT_EQ(Code::Decode(I{33}), (V{1, 0, 2}));
  EXPECT_EQ(Code::Decode(I{34}), (V{0, 1, 2}));
  EXPECT_EQ(Code::Decode(I{35}), (V{1, 1, 2}));
  EXPECT_EQ(Code::Decode(I{36}), (V{0, 0, 3}));
  EXPECT_EQ(Code::Decode(I{37}), (V{1, 0, 3}));
  EXPECT_EQ(Code::Decode(I{38}), (V{0, 1, 3}));
  EXPECT_EQ(Code::Decode(I{39}), (V{1, 1, 3}));

  EXPECT_EQ(Code::Encode(V{2, 0, 2}), 40);
  EXPECT_EQ(Code::Encode(V{3, 0, 2}), 41);
  EXPECT_EQ(Code::Encode(V{2, 1, 2}), 42);
  EXPECT_EQ(Code::Encode(V{3, 1, 2}), 43);
  EXPECT_EQ(Code::Encode(V{2, 0, 3}), 44);
  EXPECT_EQ(Code::Encode(V{3, 0, 3}), 45);
  EXPECT_EQ(Code::Encode(V{2, 1, 3}), 46);
  EXPECT_EQ(Code::Encode(V{3, 1, 3}), 47);

  EXPECT_EQ(Code::Decode(I{40}), (V{2, 0, 2}));
  EXPECT_EQ(Code::Decode(I{41}), (V{3, 0, 2}));
  EXPECT_EQ(Code::Decode(I{42}), (V{2, 1, 2}));
  EXPECT_EQ(Code::Decode(I{43}), (V{3, 1, 2}));
  EXPECT_EQ(Code::Decode(I{44}), (V{2, 0, 3}));
  EXPECT_EQ(Code::Decode(I{45}), (V{3, 0, 3}));
  EXPECT_EQ(Code::Decode(I{46}), (V{2, 1, 3}));
  EXPECT_EQ(Code::Decode(I{47}), (V{3, 1, 3}));

  EXPECT_EQ(Code::Encode(V{0, 2, 2}), 48);
  EXPECT_EQ(Code::Encode(V{1, 2, 2}), 49);
  EXPECT_EQ(Code::Encode(V{0, 3, 2}), 50);
  EXPECT_EQ(Code::Encode(V{1, 3, 2}), 51);
  EXPECT_EQ(Code::Encode(V{0, 2, 3}), 52);
  EXPECT_EQ(Code::Encode(V{1, 2, 3}), 53);
  EXPECT_EQ(Code::Encode(V{0, 3, 3}), 54);
  EXPECT_EQ(Code::Encode(V{1, 3, 3}), 55);

  EXPECT_EQ(Code::Decode(I{48}), (V{0, 2, 2}));
  EXPECT_EQ(Code::Decode(I{49}), (V{1, 2, 2}));
  EXPECT_EQ(Code::Decode(I{50}), (V{0, 3, 2}));
  EXPECT_EQ(Code::Decode(I{51}), (V{1, 3, 2}));
  EXPECT_EQ(Code::Decode(I{52}), (V{0, 2, 3}));
  EXPECT_EQ(Code::Decode(I{53}), (V{1, 2, 3}));
  EXPECT_EQ(Code::Decode(I{54}), (V{0, 3, 3}));
  EXPECT_EQ(Code::Decode(I{55}), (V{1, 3, 3}));

  EXPECT_EQ(Code::Encode(V{2, 2, 2}), 56);
  EXPECT_EQ(Code::Encode(V{3, 2, 2}), 57);
  EXPECT_EQ(Code::Encode(V{2, 3, 2}), 58);
  EXPECT_EQ(Code::Encode(V{3, 3, 2}), 59);
  EXPECT_EQ(Code::Encode(V{2, 2, 3}), 60);
  EXPECT_EQ(Code::Encode(V{3, 2, 3}), 61);
  EXPECT_EQ(Code::Encode(V{2, 3, 3}), 62);
  EXPECT_EQ(Code::Encode(V{3, 3, 3}), 63);

  EXPECT_EQ(Code::Decode(I{56}), (V{2, 2, 2}));
  EXPECT_EQ(Code::Decode(I{57}), (V{3, 2, 2}));
  EXPECT_EQ(Code::Decode(I{58}), (V{2, 3, 2}));
  EXPECT_EQ(Code::Decode(I{59}), (V{3, 3, 2}));
  EXPECT_EQ(Code::Decode(I{60}), (V{2, 2, 3}));
  EXPECT_EQ(Code::Decode(I{61}), (V{3, 2, 3}));
  EXPECT_EQ(Code::Decode(I{62}), (V{2, 3, 3}));
  EXPECT_EQ(Code::Decode(I{63}), (V{3, 3, 3}));

  //
  //
  //
  EXPECT_EQ(Code::Encode(V{4, 0, 0}), 64);
  EXPECT_EQ(Code::Encode(V{5, 0, 0}), 65);
  EXPECT_EQ(Code::Encode(V{4, 1, 0}), 66);
  EXPECT_EQ(Code::Encode(V{5, 1, 0}), 67);
  EXPECT_EQ(Code::Encode(V{4, 0, 1}), 68);
  EXPECT_EQ(Code::Encode(V{5, 0, 1}), 69);
  EXPECT_EQ(Code::Encode(V{4, 1, 1}), 70);
  EXPECT_EQ(Code::Encode(V{5, 1, 1}), 71);

  EXPECT_EQ(Code::Decode(I{64}), (V{4, 0, 0}));
  EXPECT_EQ(Code::Decode(I{65}), (V{5, 0, 0}));
  EXPECT_EQ(Code::Decode(I{66}), (V{4, 1, 0}));
  EXPECT_EQ(Code::Decode(I{67}), (V{5, 1, 0}));
  EXPECT_EQ(Code::Decode(I{68}), (V{4, 0, 1}));
  EXPECT_EQ(Code::Decode(I{69}), (V{5, 0, 1}));
  EXPECT_EQ(Code::Decode(I{70}), (V{4, 1, 1}));
  EXPECT_EQ(Code::Decode(I{71}), (V{5, 1, 1}));

  //
  //
  //
  EXPECT_EQ(Code::Encode(V{0x3FF, 0x3FF, 0x3FF}), 0x3FFFFFFF);
  EXPECT_EQ(Code::Decode(I{0x3FFFFFFF}), (V{0x3FF, 0x3FF, 0x3FF}));
  if constexpr (sizeof(I) == 8) {
    EXPECT_EQ(Code::Encode(V{0x1FFFFF, 0x1FFFFF, 0x1FFFFF}), 0x7FFFFFFFFFFFFFFFLLU);
    EXPECT_EQ(Code::Decode(I{0x7FFFFFFFFFFFFFFFLLU}), (V{0x1FFFFF, 0x1FFFFF, 0x1FFFFF}));
  }
}

} // namespace

//
//
//
TEST(MortonCode, Base) {
  TestMortonEncode1D<int>();
  TestMortonEncode1D<uint>();
  TestMortonEncode1D<int64>();
  TestMortonEncode1D<uint64>();

  TestMortonEncode2D<int>();
  TestMortonEncode2D<uint>();
  TestMortonEncode2D<int64>();
  TestMortonEncode2D<uint64>();

  TestMortonEncode3D<int>();
  TestMortonEncode3D<uint>();
  TestMortonEncode3D<int64>();
  TestMortonEncode3D<uint64>();
}

} // namespace ARIA
