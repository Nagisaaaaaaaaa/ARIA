#pragma once

#include "ARIA/Mosaic.h"
#include "ARIA/Property.h"
#include "ARIA/Tup.h"

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <thrust/detail/copy.h>

namespace ARIA {

namespace mosaic::detail {

// A `MosaicReference` is a proxy returned by dereferencing a "mosaic iterator".
// Suppose `it` is a "mosaic iterator", then `*it` will return a `MosaicReference`.
//
// Just like properties, several main features should be supported:
// 1. Can be implicitly cast to `T`.
// 2. Can be set by `T` or any other types which can be implicitly cast to `T`.
// 3. Operators should be automatically generated.
// 4. Can be handled by `Auto`.
//
// But we are unable to generate `MosaicReference`s with `ARIA_PROP` because:
// 1. The properties generated by `ARIA_PROP` are "non-owning".
//    They just obtain the references to the objects, not the objects themselves.
//    But `MosaicReference`s should be "owning".
//    They should obtain all the iterators to the storages.
// 2. It's still controversial about whether to
//    make properties generated by `ARIA_PROP` both copiable and movable.
//    But `MosaicReference`s should always be copiable.
//
// So, this implementation is something like a compromise:
// 1. Inherit from `PropertyBase`.
//    Automatically generate operators, satisfy `concept Property`, support `Auto`.
// 2. Some duplications of codes.
//    Similar to the implementation of `ARIA_PROP`.
template <typename TMosaic, typename TReferences>
  requires(is_mosaic_v<TMosaic>)
class MosaicReference final : public property::detail::PropertyBase<MosaicReference<TMosaic, TReferences>> {
private:
  static_assert(ValidMosaic<TMosaic>, "The mosaic definition is invalid");

  using T = typename is_mosaic<TMosaic>::T;
  using TMosaicPattern = typename is_mosaic<TMosaic>::TMosaicPattern;

  static constexpr size_t size = tuple_size_recursive_v<TMosaicPattern>;
  static_assert(size == boost::tuples::length<TReferences>::value,
                "The iterator types are inconsistent with the mosaic pattern");

public:
  ARIA_HOST_DEVICE constexpr explicit MosaicReference(const TReferences &references) : references_(references) {
    ForEach<size>([&]<auto i>() {
      static_assert(std::is_same_v<std::decay_t<decltype(Auto(get_recursive<i>(std::declval<TMosaicPattern>())))>,
                                   std::decay_t<decltype(Auto(references_.template get<i>()))>>,
                    "The iterator types are inconsistent with the mosaic pattern");
    });
  }

  ARIA_COPY_MOVE_ABILITY(MosaicReference, default, default);

public:
  ARIA_HOST_DEVICE constexpr T value() const {
    TMosaicPattern mosaicPattern;
    ForEach<size>([&]<auto i>() { get_recursive<i>(mosaicPattern) = references_.template get<i>(); });
    return TMosaic{}(mosaicPattern);
  }

  ARIA_HOST_DEVICE constexpr operator T() const { return value(); }

  template <typename U>
  ARIA_HOST_DEVICE constexpr MosaicReference &operator=(U &&arg) {
    T value = std::forward<U>(arg);
    TMosaicPattern mosaicPattern = TMosaic{}(value);
    ForEach<size>([&]<auto i>() { references_.template get<i>() = get_recursive<i>(mosaicPattern); });
    return *this;
  }

  template <typename U, size_t n>
  ARIA_HOST_DEVICE constexpr MosaicReference &operator=(const U (&args)[n]) {
    operator=(property::detail::ConstructWithArray<T>(args, std::make_index_sequence<n>{}));
    return *this;
  }

private:
  TReferences references_;
};

//
//
//
// \brief Generate a "mosaic iterator" with a tuple of iterators which
// are consistent with the mosaic pattern.
//
// \example ```cpp
// using T = Tup<int, float>;
// using TMosaic = Mosaic<T, MosaicPattern>;
//
// std::vector<int> is = {0, 1, 2, 3, 4};
// std::array<float, 5> fs = {0.1F, 1.2F, 2.3F, 3.4F, 4.5F};
//
// auto begin = Auto(make_mosaic_iterator<TMosaic>(Tup{is.begin(), fs.begin()}));
// *begin += T{10, 10.01F};
// T value = *begin; // The value will be `{10, 10.11F}`.
// ```
template <typename TMosaic, typename... TIterators> // For `Mosaic`, wrap the iterators with the help of `boost`.
  requires(is_mosaic_v<TMosaic>)
ARIA_HOST_DEVICE static constexpr auto make_mosaic_iterator(const Tup<TIterators...> &iterators) {
  boost::tuple<TIterators...> iteratorsBoost;
  ForEach<sizeof...(TIterators)>([&]<auto i>() { get<i>(iteratorsBoost) = get<i>(iterators); });

  return boost::make_transform_iterator(boost::make_zip_iterator(iteratorsBoost),
                                        []<typename TReferences>(const TReferences &references) {
    // Suppose `it` is a "mosaic iterator", then `*it` will return a `MosaicReference`.
    return MosaicReference<TMosaic, TReferences>{references};
  });
}

template <typename TNonMosaic, typename TIterator> // For non-`Mosaic`, simply return the unique iterator.
  requires(!is_mosaic_v<TNonMosaic>)
ARIA_HOST_DEVICE static constexpr auto make_mosaic_iterator(const Tup<TIterator> &iterator) {
  return get<0>(iterator);
}

//
//
//
// "Pointers" can be regarded as a subset of "iterators".
template <typename T, typename... TPointers>
ARIA_HOST_DEVICE static constexpr auto make_mosaic_pointer(const Tup<TPointers...> &pointers) {
  return make_mosaic_iterator<T>(pointers);
}

//
//
//
//
//
// Cast `boost::tuples::tuple` to `Tup`.
template <typename... Ts>
consteval auto BoostTuple2Tup(const boost::tuples::tuple<Ts...> &) {
  using TArrayWithNullTypes = MakeTypeArray<Ts...>;
  using TArray = TArrayWithNullTypes::template Remove<boost::tuples::null_type>;
  return to_tup_t<TArray>{};
}

template <typename TBoostTuple>
using boost_tuple_2_tup_t = decltype(BoostTuple2Tup(std::declval<TBoostTuple>()));

// Cast "mosaic iterator" to `Tup`.
template <typename TMosaicIterator>
using mosaic_iterator_2_tup_t =
    boost_tuple_2_tup_t<decltype(std::declval<TMosaicIterator>().base().get_iterator_tuple())>;

// Cast "mosaic pointer" to `Tup`.
template <typename TMosaicPointer>
using mosaic_pointer_2_tup_t =
    boost_tuple_2_tup_t<decltype(std::declval<TMosaicPointer>().base().get_iterator_tuple())>;

// Cast "mosaic iterator" to `Tup`.
template <typename TMosaicIterator>
static constexpr auto MosaicIterator2Tup(const TMosaicIterator &iterator) {
  using TTup = mosaic_iterator_2_tup_t<TMosaicIterator>;

  TTup res;
  auto iteratorsBoost = iterator.base().get_iterator_tuple();
  ForEach<rank_v<TTup>>([&]<auto i>() { get<i>(res) = get<i>(iteratorsBoost); });
  return res;
}

// Cast "mosaic pointer" to `Tup`.
template <typename TMosaicPointer>
static constexpr auto MosaicPointer2Tup(const TMosaicPointer &pointer) {
  using TTup = mosaic_pointer_2_tup_t<TMosaicPointer>;

  TTup res;
  auto pointersBoost = pointer.base().get_iterator_tuple();
  ForEach<rank_v<TTup>>([&]<auto i>() { get<i>(res) = get<i>(pointersBoost); });
  return res;
}

//
//
//
// Implementation of `copy` for "mosaic iterators".
template <typename TItIn, typename TItOut>
TItOut copy_mosaic(TItIn srcBegin, TItIn srcEnd, TItOut dst) {
  auto srcBeginTup = Auto(MosaicIterator2Tup(srcBegin));
  auto srcEndTup = Auto(MosaicIterator2Tup(srcEnd));
  auto dstTup = Auto(MosaicIterator2Tup(dst));

  using TSrcBeginTup = decltype(srcBeginTup);
  using TSrcEndTup = decltype(srcEndTup);
  using TDstTup = decltype(dstTup);

  constexpr uint rank = rank_v<TDstTup>;
  static_assert(rank_v<TSrcBeginTup> == rank && rank_v<TSrcEndTup> == rank, "Inconsistent ranks of mosaic iterators");

  TDstTup resTup;
  ForEach<rank>(
      [&]<auto i>() { get<i>(resTup) = thrust::copy(get<i>(srcBeginTup), get<i>(srcEndTup), get<i>(dstTup)); });

  return make_mosaic_iterator<...>(resTup);
}

} // namespace mosaic::detail

} // namespace ARIA
