#include "ARIA/TypeArray.h"

#include <gtest/gtest.h>

#include <vector>

namespace ARIA {

//! Forbid clang-format for this file because this file is extremely weird.
// clang-format off

namespace {

class A {
  int a;
};

class B : public A {
  int b;
};

enum class X : unsigned {
  a = 0, b, c, size
};

} // namespace



TEST(TypeArray, Base) {
  using ts0 = TypeArray<>;
  using ts1 = TypeArray<const A, volatile B&, X&&>;

  // Non array type
  {
//    using ts = TypeArray<ts0>;
//    using ts = TypeArray<const A, ts1>;
//    using ts = TypeArray<ts1, const A>;
  }

  // Size
  {
    static_assert(ts0::size == 0);
    static_assert(ts1::size == 3);
  }

  // Make type array
  {
    // Empty
    static_assert(std::is_same_v<MakeTypeArray<>, TypeArray<>>);
    static_assert(std::is_same_v<MakeTypeArray<TypeArray<>>, TypeArray<>>);
    static_assert(std::is_same_v<MakeTypeArray<TypeArray<>, TypeArray<>>, TypeArray<>>);

    // Non array types
    static_assert(std::is_same_v<MakeTypeArray<const A>, TypeArray<const A>>);
    static_assert(std::is_same_v<MakeTypeArray<const A, volatile B&>, TypeArray<const A, volatile B&>>);

    // Array types
    static_assert(std::is_same_v<MakeTypeArray<TypeArray<const A>>, TypeArray<const A>>);
    static_assert(std::is_same_v<MakeTypeArray<TypeArray<const A>, TypeArray<volatile B&, X&&>>, TypeArray<const A, volatile B&, X&&>>);

    // Hybrid
    static_assert(std::is_same_v<MakeTypeArray<const A, TypeArray<volatile B&, X&&>>, TypeArray<const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<MakeTypeArray<const A, volatile B&, TypeArray<X&&>>, TypeArray<const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<MakeTypeArray<TypeArray<const A>, volatile B&, X&&>, TypeArray<const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<MakeTypeArray<TypeArray<const A, volatile B&>, X&&>, TypeArray<const A, volatile B&, X&&>>);
  }

  // Get
  {
    // Empty
//    using t = ts0::Get<-1>;
//    using t = ts0::Get<0>;
//    using t = ts0::Get<1>;

    // Non empty
//    using t = ts1::Get<-4>;
    static_assert(std::is_same_v<ts1::Get<-3>, const A>);
    static_assert(std::is_same_v<ts1::Get<-2>, volatile B&>);
    static_assert(std::is_same_v<ts1::Get<-1>, X&&>);
    static_assert(std::is_same_v<ts1::Get<0>, const A>);
    static_assert(std::is_same_v<ts1::Get<1>, volatile B&>);
    static_assert(std::is_same_v<ts1::Get<2>, X&&>);
//    using t = ts1::Get<3>;
  }

  // Slice
  {
    using ts2 = TypeArray<const A, volatile B&, X&&, A, B, X>;

    // Zero step
//    using t = ts0::Slice<0, 1, 0>;
//    using t = ts1::Slice<0, 1, 0>;

    // Empty
    static_assert(std::is_same_v<ts0::Slice<0, 0, 1>, TypeArray<>>);
    static_assert(std::is_same_v<ts0::Slice<0, 1, 1>, TypeArray<>>);
    static_assert(std::is_same_v<ts0::Slice<1, 0, 1>, TypeArray<>>);
    static_assert(std::is_same_v<ts0::Slice<1, 1, 1>, TypeArray<>>);

    // Positive begin and end, step = 1
    static_assert(std::is_same_v<ts1::Slice<0, 0, 1>, TypeArray<>>);
    static_assert(std::is_same_v<ts1::Slice<0, 1, 1>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts1::Slice<0, 2, 1>, TypeArray<const A, volatile B&>>);
    static_assert(std::is_same_v<ts1::Slice<0, 3, 1>, TypeArray<const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Slice<1, 3, 1>, TypeArray<volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Slice<2, 3, 1>, TypeArray<X&&>>);
    static_assert(std::is_same_v<ts1::Slice<3, 3, 1>, TypeArray<>>);

    // Positive begin and end, step = -1
    static_assert(std::is_same_v<ts1::Slice<-4, -4, -1>, TypeArray<>>);
    static_assert(std::is_same_v<ts1::Slice< 0, -4, -1>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts1::Slice< 1, -4, -1>, TypeArray<volatile B&, const A>>);
    static_assert(std::is_same_v<ts1::Slice< 2, -4, -1>, TypeArray<X&&, volatile B&, const A>>);
    static_assert(std::is_same_v<ts1::Slice< 2,  0, -1>, TypeArray<X&&, volatile B&>>);
    static_assert(std::is_same_v<ts1::Slice< 2,  1, -1>, TypeArray<X&&>>);
    static_assert(std::is_same_v<ts1::Slice< 2,  2, -1>, TypeArray<>>);

    // Positive begin and end, step = 2
    static_assert(std::is_same_v<ts1::Slice<0, 0, 2>, TypeArray<>>);
    static_assert(std::is_same_v<ts1::Slice<0, 1, 2>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts1::Slice<0, 2, 2>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts1::Slice<0, 3, 2>, TypeArray<const A, X&&>>);
    static_assert(std::is_same_v<ts1::Slice<1, 3, 2>, TypeArray<volatile B&>>);
    static_assert(std::is_same_v<ts1::Slice<2, 3, 2>, TypeArray<X&&>>);
    static_assert(std::is_same_v<ts1::Slice<3, 3, 2>, TypeArray<>>);

    // Positive begin and end, step = -2
    static_assert(std::is_same_v<ts1::Slice<-4, -4, -2>, TypeArray<>>);
    static_assert(std::is_same_v<ts1::Slice< 0, -4, -2>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts1::Slice< 1, -4, -2>, TypeArray<volatile B&>>);
    static_assert(std::is_same_v<ts1::Slice< 2, -4, -2>, TypeArray<X&&, const A>>);
    static_assert(std::is_same_v<ts1::Slice< 2,  0, -2>, TypeArray<X&&>>);
    static_assert(std::is_same_v<ts1::Slice< 2,  1, -2>, TypeArray<X&&>>);
    static_assert(std::is_same_v<ts1::Slice< 2,  2, -2>, TypeArray<>>);

    // Positive begin and end, step = 3
    static_assert(std::is_same_v<ts2::Slice<0, 0, 3>, TypeArray<>>);
    static_assert(std::is_same_v<ts2::Slice<0, 1, 3>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts2::Slice<0, 2, 3>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts2::Slice<0, 3, 3>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts2::Slice<0, 4, 3>, TypeArray<const A, A>>);
    static_assert(std::is_same_v<ts2::Slice<0, 5, 3>, TypeArray<const A, A>>);
    static_assert(std::is_same_v<ts2::Slice<0, 6, 3>, TypeArray<const A, A>>);
    static_assert(std::is_same_v<ts2::Slice<1, 6, 3>, TypeArray<volatile B&, B>>);
    static_assert(std::is_same_v<ts2::Slice<2, 6, 3>, TypeArray<X&&, X>>);
    static_assert(std::is_same_v<ts2::Slice<3, 6, 3>, TypeArray<A>>);
    static_assert(std::is_same_v<ts2::Slice<4, 6, 3>, TypeArray<B>>);
    static_assert(std::is_same_v<ts2::Slice<5, 6, 3>, TypeArray<X>>);
    static_assert(std::is_same_v<ts2::Slice<6, 6, 3>, TypeArray<>>);

    // Positive begin and end, step = -3
    static_assert(std::is_same_v<ts2::Slice<-7, -7, -3>, TypeArray<>>);
    static_assert(std::is_same_v<ts2::Slice< 0, -7, -3>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts2::Slice< 1, -7, -3>, TypeArray<volatile B&>>);
    static_assert(std::is_same_v<ts2::Slice< 2, -7, -3>, TypeArray<X&&>>);
    static_assert(std::is_same_v<ts2::Slice< 3, -7, -3>, TypeArray<A, const A>>);
    static_assert(std::is_same_v<ts2::Slice< 4, -7, -3>, TypeArray<B, volatile B&>>);
    static_assert(std::is_same_v<ts2::Slice< 5, -7, -3>, TypeArray<X, X&&>>);
    static_assert(std::is_same_v<ts2::Slice< 5,  0, -3>, TypeArray<X, X&&>>);
    static_assert(std::is_same_v<ts2::Slice< 5,  1, -3>, TypeArray<X, X&&>>);
    static_assert(std::is_same_v<ts2::Slice< 5,  2, -3>, TypeArray<X>>);
    static_assert(std::is_same_v<ts2::Slice< 5,  3, -3>, TypeArray<X>>);
    static_assert(std::is_same_v<ts2::Slice< 5,  4, -3>, TypeArray<X>>);
    static_assert(std::is_same_v<ts2::Slice< 5,  5, -3>, TypeArray<>>);

    // Negative begin and end, positive step
    for_each::detail::ForEachImpl<0, 6, 1>([&] <auto begin_pos> {
      for_each::detail::ForEachImpl<0, 6, 1>([&] <auto end_pos> {
        for_each::detail::ForEachImpl<1, 10, 1>([&] <auto step> {
          constexpr int begin_neg = begin_pos - 6;
          constexpr int end_neg = end_pos - 6;

          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_pos, end_neg, step>>);
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_neg, end_pos, step>>);
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_neg, end_neg, step>>);
        });
        for_each::detail::ForEachImpl<-10, 0, 1>([&] <auto step> {
          constexpr int begin_neg = begin_pos - 6;
          constexpr int end_neg = end_pos - 6;

          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_pos, end_neg, step>>);
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_neg, end_pos, step>>);
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_neg, end_neg, step>>);
        });
      });
    });

    // Positive begin overflow
    for_each::detail::ForEachImpl<5, 10, 1>([&] <auto begin_pos> {
      for_each::detail::ForEachImpl<0, 6, 1>([&] <auto end_pos> {
        for_each::detail::ForEachImpl<1, 10, 1>([&] <auto step> {
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<5, end_pos, step>>);
        });
        for_each::detail::ForEachImpl<-10, 0, 1>([&] <auto step> {
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<5, end_pos, step>>);
        });
      });
    });

    // Negative begin overflow
    for_each::detail::ForEachImpl<-10, -5, 1>([&] <auto begin_pos> {
      for_each::detail::ForEachImpl<0, 6, 1>([&] <auto end_pos> {
        for_each::detail::ForEachImpl<1, 10, 1>([&] <auto step> {
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<0, end_pos, step>>);
        });
        for_each::detail::ForEachImpl<-10, 0, 1>([&] <auto step> {
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<0, end_pos, step>>);
        });
      });
    });

    // Positive end overflow
    for_each::detail::ForEachImpl<0, 6, 1>([&] <auto begin_pos> {
      for_each::detail::ForEachImpl<6, 10, 1>([&] <auto end_pos> {
        for_each::detail::ForEachImpl<1, 10, 1>([&] <auto step> {
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_pos, 6, step>>);
        });
        for_each::detail::ForEachImpl<-10, 0, 1>([&] <auto step> {
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_pos, 6, step>>);
        });
      });
    });

    // Negative end overflow
    for_each::detail::ForEachImpl<0, 6, 1>([&] <auto begin_pos> {
      for_each::detail::ForEachImpl<-10, -6, 1>([&] <auto end_pos> {
        for_each::detail::ForEachImpl<1, 10, 1>([&] <auto step> {
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_pos, -7, step>>);
        });
        for_each::detail::ForEachImpl<-10, 0, 1>([&] <auto step> {
          static_assert(std::is_same_v<ts2::Slice<begin_pos, end_pos, step>, ts2::Slice<begin_pos, -7, step>>);
        });
      });
    });
  }

  // Reverse
  {
    // Empty
    static_assert(std::is_same_v<ts0::Reverse<>, TypeArray<>>);

    // Non empty
    static_assert(std::is_same_v<ts1::Reverse<>, TypeArray<X&&, volatile B&, const A>>);
  }

  // Erase
  {
    // Empty
//    using ts = ts0::Erase<0>;
//    using ts = ts0::Erase<1>;
//    using ts = ts0::Erase<-1>;

    // Non empty
//    using ts = ts1::Erase<-4>;
    static_assert(std::is_same_v<ts1::Erase<-3>, TypeArray<volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Erase<-2>, TypeArray<const A, X&&>>);
    static_assert(std::is_same_v<ts1::Erase<-1>, TypeArray<const A, volatile B&>>);
    static_assert(std::is_same_v<ts1::Erase<0>, TypeArray<volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Erase<1>, TypeArray<const A, X&&>>);
    static_assert(std::is_same_v<ts1::Erase<2>, TypeArray<const A, volatile B&>>);
//    using ts = ts1::Erase<3>;

    // Erase until empty
    static_assert(std::is_same_v<ts1::Erase<0>::Erase<0>::Erase<0>, TypeArray<>>);
  }

  // Insert
  {
    // Empty
//    using ts = ts0::Insert<0, void>;

    // Non empty, non array type
//    using ts = ts1::Insert<-4, void>;
    static_assert(std::is_same_v<ts1::Insert<-3, void>, TypeArray<void, const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<-2, void>, TypeArray<const A, void, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<-1, void>, TypeArray<const A, volatile B&, void, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<0, void>, TypeArray<void, const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<1, void>, TypeArray<const A, void, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<2, void>, TypeArray<const A, volatile B&, void, X&&>>);
//    using ts = ts1::Insert<3, void>;

    // Non empty, array type
    static_assert(std::is_same_v<ts1::Insert<-3, TypeArray<void, void*>>, TypeArray<void, void*, const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<-2, TypeArray<void, void*>>, TypeArray<const A, void, void*, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<-1, TypeArray<void, void*>>, TypeArray<const A, volatile B&, void, void*, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<0, TypeArray<void, void*>>, TypeArray<void, void*, const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<1, TypeArray<void, void*>>, TypeArray<const A, void, void*, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Insert<2, TypeArray<void, void*>>, TypeArray<const A, volatile B&, void, void*, X&&>>);
  }

  // Pop
  {
    // Empty
//    using ts = ts0::PopFront<>;
//    using ts = ts0::PopBack<>;

    // Non empty
    static_assert(std::is_same_v<ts1::PopFront<>, TypeArray<volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::PopBack<>, TypeArray<const A, volatile B&>>);
  }

  // Push
  {
    // Empty
    static_assert(std::is_same_v<ts0::PushFront<const A>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts0::PushFront<TypeArray<const A>>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts0::PushFront<TypeArray<const A, volatile B&>>, TypeArray<const A, volatile B&>>);

    static_assert(std::is_same_v<ts0::PushBack<const A>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts0::PushBack<TypeArray<const A>>, TypeArray<const A>>);
    static_assert(std::is_same_v<ts0::PushBack<TypeArray<const A, volatile B&>>, TypeArray<const A, volatile B&>>);

    // Non empty
    static_assert(std::is_same_v<ts1::PushFront<A>, TypeArray<A, const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::PushFront<TypeArray<A>>, TypeArray<A, const A, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::PushFront<TypeArray<A, B>>, TypeArray<A, B, const A, volatile B&, X&&>>);

    static_assert(std::is_same_v<ts1::PushBack<A>, TypeArray<const A, volatile B&, X&&, A>>);
    static_assert(std::is_same_v<ts1::PushBack<TypeArray<A>>, TypeArray<const A, volatile B&, X&&, A>>);
    static_assert(std::is_same_v<ts1::PushBack<TypeArray<A, B>>, TypeArray<const A, volatile B&, X&&, A, B>>);
  }

  // N of
  {
    using ts2 = TypeArray<A, A, B, X, X, B, A, B, B, A, X, X>;

    // Array type
//    ts1::nOf<TypeArray<const A>>;

    // Empty
    static_assert(ts0::nOf<void> == 0);
    static_assert(ts0::nOf<A> == 0);

    // Non empty, n == 1
    static_assert(ts1::nOf<void> == 0);
    static_assert(ts1::nOf<A> == 0);
    static_assert(ts1::nOf<const A> == 1);
    static_assert(ts1::nOf<volatile B&> == 1);
    static_assert(ts1::nOf<X&&> == 1);

    // Non empty, n > 1
    static_assert(ts2::nOf<A> == 4);
    static_assert(ts2::nOf<B> == 4);
    static_assert(ts2::nOf<X> == 4);
  }

  // Has
  {
    // Array type
//    ts1::has<TypeArray<const A>>;

    // Empty
    static_assert(!ts0::has<void>);
    static_assert(!ts0::has<A>);

    // Non empty
    static_assert(!ts1::has<void>);
    static_assert(!ts1::has<A>);
    static_assert(ts1::has<const A>);
    static_assert(ts1::has<volatile B&>);
    static_assert(ts1::has<X&&>);
  }

  // Idx
  {
    using ts2 = TypeArray<A, A, B, X, X, B, A, B, B, A, X, X>;

    // Empty
//    ts0::firstIdx<void>;
//    ts0::firstIdx<A>;

//    ts0::lastIdx<void>;
//    ts0::lastIdx<A>;

    // Non empty
    static_assert(ts2::firstIdx<A> == 0);
    static_assert(ts2::firstIdx<B> == 2);
    static_assert(ts2::firstIdx<X> == 3);

    static_assert(ts2::lastIdx<A> == 9);
    static_assert(ts2::lastIdx<B> == 8);
    static_assert(ts2::lastIdx<X> == 11);
  }

  // Remove
  {
    using ts2 = TypeArray<A, A, B, X, X, B, A, B, B, A, X, X>;

    // Empty
    static_assert(std::is_same_v<ts0::Remove<void>, ts0>);
    static_assert(std::is_same_v<ts0::Remove<int>, ts0>);

    // Remove 1
    static_assert(std::is_same_v<ts1::Remove<void>, ts1>);
    static_assert(std::is_same_v<ts1::Remove<int>, ts1>);
    static_assert(std::is_same_v<ts1::Remove<const A>, TypeArray<volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Remove<volatile B&>, TypeArray<const A, X&&>>);
    static_assert(std::is_same_v<ts1::Remove<X&&>, TypeArray<const A, volatile B&>>);

    // Remove > 1
    static_assert(std::is_same_v<ts2::Remove<A>, TypeArray<B, X, X, B, B, B, X, X>>);
    static_assert(std::is_same_v<ts2::Remove<B>, TypeArray<A, A, X, X, A, A, X, X>>);
    static_assert(std::is_same_v<ts2::Remove<X>, TypeArray<A, A, B, B, A, B, B, A>>);
  }

  // Replace
  {
    using ts2 = TypeArray<A, A, B, X, X, B, A, B, B, A, X, X>;
    struct D {};
    struct E : D {};

    // Empty
    static_assert(std::is_same_v<ts0::Replace<void, int>, TypeArray<>>);

    // Non empty, non array type
    static_assert(std::is_same_v<ts1::Replace<const A, volatile A&>, TypeArray<volatile A&, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Replace<volatile B&, B&&>, TypeArray<const A, B&&, X&&>>);
    static_assert(std::is_same_v<ts1::Replace<X&&, const A>, TypeArray<const A, volatile B&, const A>>);

    static_assert(std::is_same_v<ts2::Replace<A, D>, TypeArray<D, D, B, X, X, B, D, B, B, D, X, X>>);
    static_assert(std::is_same_v<ts2::Replace<B, D>, TypeArray<A, A, D, X, X, D, A, D, D, A, X, X>>);
    static_assert(std::is_same_v<ts2::Replace<X, D>, TypeArray<A, A, B, D, D, B, A, B, B, A, D, D>>);

    // Non empty, array type
    static_assert(std::is_same_v<ts1::Replace<const A, TypeArray<volatile A&, A&&>>, TypeArray<volatile A&, A&&, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Replace<volatile B&, TypeArray<B&&, const B>>, TypeArray<const A, B&&, const B, X&&>>);
    static_assert(std::is_same_v<ts1::Replace<X&&, TypeArray<const X, volatile X&>>, TypeArray<const A, volatile B&, const X, volatile X&>>);

    static_assert(std::is_same_v<ts2::Replace<A, TypeArray<D, E>>, TypeArray<D, E, D, E, B, X, X, B, D, E, B, B, D, E, X, X>>);
    static_assert(std::is_same_v<ts2::Replace<B, TypeArray<D, E>>, TypeArray<A, A, D, E, X, X, D, E, A, D, E, D, E, A, X, X>>);
    static_assert(std::is_same_v<ts2::Replace<X, TypeArray<D, E>>, TypeArray<A, A, B, D, E, D, E, B, A, B, B, A, D, E, D, E>>);
  }

  // For each
  {
    // Empty
    static_assert(std::is_same_v<ts0::ForEach<std::decay_t>, TypeArray<>>);

    // Non empty
    static_assert(std::is_same_v<ts1::ForEach<std::remove_const_t>, TypeArray<A, volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::ForEach<std::remove_reference_t>, TypeArray<const A, volatile B, X>>);
    static_assert(std::is_same_v<ts1::ForEach<std::decay_t>, TypeArray<A, B, X>>);
  }

  // Filter
  {
    // Empty
    static_assert(std::is_same_v<ts0::Filter<std::is_reference>, TypeArray<>>);

    // Non empty
    static_assert(std::is_same_v<ts1::Filter<std::is_reference>, TypeArray<volatile B&, X&&>>);
    static_assert(std::is_same_v<ts1::Filter<std::is_const>, TypeArray<const A>>);
  }

  // ForEach
  {
    std::stringstream ss;

    using ts = MakeTypeArray<const int, volatile float&, void>;
    ForEach<ts>([&] <typename T> {
      if constexpr (std::is_same_v<T, const int>)
        ss << "const int ";
      else if constexpr (std::is_same_v<T, volatile float&>)
        ss << "volatile float& ";
      else if constexpr (std::is_same_v<T, void>)
        ss << "void ";
      else
        ss << "unknown";
    });

    EXPECT_EQ(ss.str(), "const int volatile float& void ");
  }
}

// clang-format on

} // namespace ARIA
