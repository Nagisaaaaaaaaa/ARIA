#include "ARIA/TypeSet.h"
#include "ARIA/ForEach.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(TypeSet, Base) {
  {
    // `ArrayType`.
    static_assert(type_array::detail::ArrayType<MakeTypeSet<>>);
    static_assert(type_array::detail::ArrayType<MakeTypeSet<void>>);
    static_assert(type_array::detail::ArrayType<MakeTypeSet<int, float, double>>);
    // static_assert(type_array::detail::ArrayType<MakeTypeSet<int, float, float>>);

    // `MakeTypeSet`.
    static_assert(std::is_same_v<MakeTypeSet<MakeTypeArray<>>, TypeSet<>>);
    static_assert(std::is_same_v<MakeTypeSet<MakeTypeArray<void>>, TypeSet<void>>);
    static_assert(std::is_same_v<MakeTypeSet<MakeTypeArray<int, float, double>>, TypeSet<int, float, double>>);
    static_assert(std::is_same_v<MakeTypeSet<MakeTypeArray<int, float>, double>, TypeSet<int, float, double>>);

    // `size`.
    static_assert(MakeTypeSet<>::size == 0);
    static_assert(MakeTypeSet<void>::size == 1);
    static_assert(MakeTypeSet<int, float, double>::size == 3);

    // `nOf`.
    static_assert(MakeTypeSet<>::nOf<void> == 0);
    static_assert(MakeTypeSet<void>::nOf<void> == 1);
    static_assert(MakeTypeSet<void>::nOf<int> == 0);
    static_assert(MakeTypeSet<int, float, double>::nOf<int> == 1);
    static_assert(MakeTypeSet<int, float, double>::nOf<float> == 1);
    static_assert(MakeTypeSet<int, float, double>::nOf<double> == 1);
    static_assert(MakeTypeSet<int, float, double>::nOf<void> == 0);

    // `has`.
    static_assert(!MakeTypeSet<>::has<void>);
    static_assert(MakeTypeSet<void>::has<void>);
    static_assert(!MakeTypeSet<void>::has<int>);
    static_assert(MakeTypeSet<int, float, double>::has<int>);
    static_assert(MakeTypeSet<int, float, double>::has<float>);
    static_assert(MakeTypeSet<int, float, double>::has<double>);
    static_assert(!MakeTypeSet<int, float, double>::has<void>);
  }

  // `Get`, `idx`.
  {
    using ts = MakeTypeSet<int8, int16, int, int64, const uint8, uint16 &, const uint &, uint64 &&>;

    static_assert(std::is_same_v<ts::Get<0>, int8>);
    static_assert(std::is_same_v<ts::Get<1>, int16>);
    static_assert(std::is_same_v<ts::Get<2>, int>);
    static_assert(std::is_same_v<ts::Get<3>, int64>);
    static_assert(std::is_same_v<ts::Get<4>, const uint8>);
    static_assert(std::is_same_v<ts::Get<5>, uint16 &>);
    static_assert(std::is_same_v<ts::Get<6>, const uint &>);
    static_assert(std::is_same_v<ts::Get<7>, uint64 &&>);

    static_assert(ts::idx<int8> == 0);
    static_assert(ts::idx<int16> == 1);
    static_assert(ts::idx<int> == 2);
    static_assert(ts::idx<int64> == 3);
    static_assert(ts::idx<const uint8> == 4);
    static_assert(ts::idx<uint16 &> == 5);
    static_assert(ts::idx<const uint &> == 6);
    static_assert(ts::idx<uint64 &&> == 7);
  }

  {
    using ts = MakeTypeSet<int, const int, volatile int,       //
                           int &, const int &, volatile int &, //
                           int &&, const int &&, volatile int &&>;

    static_assert(std::is_same_v<ts::Get<0>, int>);
    static_assert(std::is_same_v<ts::Get<1>, const int>);
    static_assert(std::is_same_v<ts::Get<2>, volatile int>);
    static_assert(std::is_same_v<ts::Get<3>, int &>);
    static_assert(std::is_same_v<ts::Get<4>, const int &>);
    static_assert(std::is_same_v<ts::Get<5>, volatile int &>);
    static_assert(std::is_same_v<ts::Get<6>, int &&>);
    static_assert(std::is_same_v<ts::Get<7>, const int &&>);
    static_assert(std::is_same_v<ts::Get<8>, volatile int &&>);

    static_assert(ts::idx<int> == 0);
    static_assert(ts::idx<const int> == 1);
    static_assert(ts::idx<volatile int> == 2);
    static_assert(ts::idx<int &> == 3);
    static_assert(ts::idx<const int &> == 4);
    static_assert(ts::idx<volatile int &> == 5);
    static_assert(ts::idx<int &&> == 6);
    static_assert(ts::idx<const int &&> == 7);
    static_assert(ts::idx<volatile int &&> == 8);
  }
}

TEST(TypeSet, LargeScale) {
  using ts = MakeTypeSet<C<0>, C<1>, C<2>, C<3>, C<4>, C<5>, C<6>, C<7>, C<8>, C<9>,                      //
                                                                                                          //
                         C<10>, C<11>, C<12>, C<13>, C<14>, C<15>, C<16>, C<17>, C<18>, C<19>,            //
                         C<20>, C<21>, C<22>, C<23>, C<24>, C<25>, C<26>, C<27>, C<28>, C<29>,            //
                         C<30>, C<31>, C<32>, C<33>, C<34>, C<35>, C<36>, C<37>, C<38>, C<39>,            //
                         C<40>, C<41>, C<42>, C<43>, C<44>, C<45>, C<46>, C<47>, C<48>, C<49>,            //
                         C<50>, C<51>, C<52>, C<53>, C<54>, C<55>, C<56>, C<57>, C<58>, C<59>,            //
                         C<60>, C<61>, C<62>, C<63>, C<64>, C<65>, C<66>, C<67>, C<68>, C<69>,            //
                         C<70>, C<71>, C<72>, C<73>, C<74>, C<75>, C<76>, C<77>, C<78>, C<79>,            //
                         C<80>, C<81>, C<82>, C<83>, C<84>, C<85>, C<86>, C<87>, C<88>, C<89>,            //
                         C<90>, C<91>, C<92>, C<93>, C<94>, C<95>, C<96>, C<97>, C<98>, C<99>,            //
                                                                                                          //
                         C<100>, C<101>, C<102>, C<103>, C<104>, C<105>, C<106>, C<107>, C<108>, C<109>,  //
                         C<110>, C<111>, C<112>, C<113>, C<114>, C<115>, C<116>, C<117>, C<118>, C<119>,  //
                         C<120>, C<121>, C<122>, C<123>, C<124>, C<125>, C<126>, C<127>, C<128>, C<129>,  //
                         C<130>, C<131>, C<132>, C<133>, C<134>, C<135>, C<136>, C<137>, C<138>, C<139>,  //
                         C<140>, C<141>, C<142>, C<143>, C<144>, C<145>, C<146>, C<147>, C<148>, C<149>,  //
                         C<150>, C<151>, C<152>, C<153>, C<154>, C<155>, C<156>, C<157>, C<158>, C<159>,  //
                         C<160>, C<161>, C<162>, C<163>, C<164>, C<165>, C<166>, C<167>, C<168>, C<169>,  //
                         C<170>, C<171>, C<172>, C<173>, C<174>, C<175>, C<176>, C<177>, C<178>, C<179>,  //
                         C<180>, C<181>, C<182>, C<183>, C<184>, C<185>, C<186>, C<187>, C<188>, C<189>,  //
                         C<190>, C<191>, C<192>, C<193>, C<194>, C<195>, C<196>, C<197>, C<198>, C<199>,  //
                                                                                                          //
                         C<200>, C<201>, C<202>, C<203>, C<204>, C<205>, C<206>, C<207>, C<208>, C<209>,  //
                         C<210>, C<211>, C<212>, C<213>, C<214>, C<215>, C<216>, C<217>, C<218>, C<219>,  //
                         C<220>, C<221>, C<222>, C<223>, C<224>, C<225>, C<226>, C<227>, C<228>, C<229>,  //
                         C<230>, C<231>, C<232>, C<233>, C<234>, C<235>, C<236>, C<237>, C<238>, C<239>,  //
                         C<240>, C<241>, C<242>, C<243>, C<244>, C<245>, C<246>, C<247>, C<248>, C<249>,  //
                         C<250>, C<251>, C<252>, C<253>, C<254>, C<255>, C<256>, C<257>, C<258>, C<259>,  //
                         C<260>, C<261>, C<262>, C<263>, C<264>, C<265>, C<266>, C<267>, C<268>, C<269>,  //
                         C<270>, C<271>, C<272>, C<273>, C<274>, C<275>, C<276>, C<277>, C<278>, C<279>,  //
                         C<280>, C<281>, C<282>, C<283>, C<284>, C<285>, C<286>, C<287>, C<288>, C<289>,  //
                         C<290>, C<291>, C<292>, C<293>, C<294>, C<295>, C<296>, C<297>, C<298>, C<299>,  //
                                                                                                          //
                         C<300>, C<301>, C<302>, C<303>, C<304>, C<305>, C<306>, C<307>, C<308>, C<309>,  //
                         C<310>, C<311>, C<312>, C<313>, C<314>, C<315>, C<316>, C<317>, C<318>, C<319>,  //
                         C<320>, C<321>, C<322>, C<323>, C<324>, C<325>, C<326>, C<327>, C<328>, C<329>,  //
                         C<330>, C<331>, C<332>, C<333>, C<334>, C<335>, C<336>, C<337>, C<338>, C<339>,  //
                         C<340>, C<341>, C<342>, C<343>, C<344>, C<345>, C<346>, C<347>, C<348>, C<349>,  //
                         C<350>, C<351>, C<352>, C<353>, C<354>, C<355>, C<356>, C<357>, C<358>, C<359>,  //
                         C<360>, C<361>, C<362>, C<363>, C<364>, C<365>, C<366>, C<367>, C<368>, C<369>,  //
                         C<370>, C<371>, C<372>, C<373>, C<374>, C<375>, C<376>, C<377>, C<378>, C<379>,  //
                         C<380>, C<381>, C<382>, C<383>, C<384>, C<385>, C<386>, C<387>, C<388>, C<389>,  //
                         C<390>, C<391>, C<392>, C<393>, C<394>, C<395>, C<396>, C<397>, C<398>, C<399>,  //
                                                                                                          //
                         C<400>, C<401>, C<402>, C<403>, C<404>, C<405>, C<406>, C<407>, C<408>, C<409>,  //
                         C<410>, C<411>, C<412>, C<413>, C<414>, C<415>, C<416>, C<417>, C<418>, C<419>,  //
                         C<420>, C<421>, C<422>, C<423>, C<424>, C<425>, C<426>, C<427>, C<428>, C<429>,  //
                         C<430>, C<431>, C<432>, C<433>, C<434>, C<435>, C<436>, C<437>, C<438>, C<439>,  //
                         C<440>, C<441>, C<442>, C<443>, C<444>, C<445>, C<446>, C<447>, C<448>, C<449>,  //
                         C<450>, C<451>, C<452>, C<453>, C<454>, C<455>, C<456>, C<457>, C<458>, C<459>,  //
                         C<460>, C<461>, C<462>, C<463>, C<464>, C<465>, C<466>, C<467>, C<468>, C<469>,  //
                         C<470>, C<471>, C<472>, C<473>, C<474>, C<475>, C<476>, C<477>, C<478>, C<479>,  //
                         C<480>, C<481>, C<482>, C<483>, C<484>, C<485>, C<486>, C<487>, C<488>, C<489>>; //

  ForEach<490>([]<auto i>() { static_assert(ts::idx<C<i>> == i); });
  ForEach<490>([]<auto i>() { static_assert(std::is_same_v<ts::Get<i>, C<i>>); });
}

} // namespace ARIA
