#pragma once

#include "ARIA/Layout.h"
#include "ARIA/TypeArray.h"

#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace ARIA {

namespace tensor_vector::detail {

//! For future developers: Please read the following comments to help you understand the codes.
//!
//! `TensorVector` should be designed as flexible and user-friendly as possible.
//! For example, you may want to declare `TensorVector`s with default layout with the following codes:
//!   TensorVector<float, SpaceHost> v;   // An 1D host vector containing `float`,
//!                                       // similar to `std::vector<float>`.
//!   TensorVectorHost<float> v;          // The same.
//!
//!   TensorVector<float, SpaceDevice> v; // An 1D device vector containing `float`,
//!                                       // similar to `thrust::device_vector<float>`.
//!   TensorVectorDevice<float> v;        // The same.
//!
//! or 2D ones:
//!   TensorVector<float, C<2>, SpaceHost> v;   // A 2D host vector containing `float`.
//!   TensorVectorHost<float, C<2>> v;          // The same.
//!   TensorVector<float, C<2>, SpaceDevice> v; // A 2D device vector containing `float`.
//!   TensorVectorDevice<float, C<2>> v;        // The same.
//!
//! or with user-defined layouts:
//!   auto myLayout = make_layout(...);
//!   using MyLayout = decltype(myLayout);
//!
//!   TensorVector<float, SpaceHost, MyLayout> v; // Use user-defined layout.
//!   TensorVectorHost<float, MyLayout> v;        // The same.
//!
//! To support all these features, we must "parse" the given template parameters.
//! The "parser" should do the following things:
//!   1. For optional parameters, add default ones if not exist,
//!   2. For must parameters, check if exist,
//!   3. Check the parameter order (eg: layouts should be given after spaces),
//!   4. Check duplication,
//!   5. Check useless parameters.
//!
//! See `TensorVectorArgs` for implementation of the parser.

using type_array::detail::ArrayType;
using type_array::detail::NonArrayType;

// Whether the given type is a rank type.
template <NonArrayType U>
static constexpr bool isRank = ConstantIntegral<U>;

// Whether the given type is a space type.
template <NonArrayType U>
static constexpr bool isSpace = std::is_same_v<U, SpaceHost> || std::is_same_v<U, SpaceDevice>;

//
//
//
// This class helps parse the template parameters given to `TensorVector<T, ...>`.
// Implementation is based on the `TypeArray`.
template <typename... RawArgs>
class TensorVectorArgs {
private:
  // See the comments below.
  template <uint n, typename... Zeros>
  static decltype(auto) MakeLayoutMajorWithNZerosImpl(Zeros &&...zeros) {
    if constexpr (n == 0)
      return make_layout_major(std::forward<Zeros>(zeros)...);
    else
      return MakeLayoutMajorWithNZerosImpl<n - 1>(0, std::forward<Zeros>(zeros)...);
  }

  // A function wrapper which calls `make_layout_major` with `n` runtime zeros, that is,
  // `make_layout_major(0, 0, ..., 0)`.
  template <uint n>
  static decltype(auto) MakeLayoutMajorWithNZeros() {
    return MakeLayoutMajorWithNZerosImpl<n>();
  }

private:
  // Push the raw args into a `TypeArray`.
  //! `Make` is necessary here, because `RawArgs` may contain `ArrayType`s,
  //! so `MakeTypeArray` will perform flattening.
  //! See the implementation of `TensorVectorHost`, which will be introduced in detail later.
  using Args = MakeTypeArray<RawArgs...>;

  //
  //
  //
  // 1. The first argument is rank, a compile-time constant integral, which is optional.
  //
  // Fetch the user-defined layouts from the arguments, and count the numbers.
  using TLayoutsFetched = Args::template Filter<is_layout>;
  static_assert(TLayoutsFetched::size <= 1, "Tensor vector should not be defined with multiple layouts");

  // If there's no user-defined layout, add a dummy rank-1 layout,
  // which will later ONLY be used to compute the default rank.
  using TLayoutsFetchedDummyed = std::conditional_t<
      TLayoutsFetched ::size == 0,
      typename TLayoutsFetched ::template PushFront<std::decay_t<decltype(make_layout_major(C<0>{}))>>,
      TLayoutsFetched>;
  // Now, `TLayoutsFetchedDummyed` contains exactly one unique layout.
  // The default rank will be defined by the unique layout.

  // Set default rank to a compile time constant.
  // If there already exist some layout, use the corresponding rank, else, use the dummy rank-1 layout.
  using DefaultRank = C<TLayoutsFetchedDummyed::template Get<0>::rank>;

  // If the type array is empty, add default rank to it.
  using ArgsRank0 = std::conditional_t<Args::size == 0, typename Args::template PushFront<DefaultRank>, Args>;

  // If the first element of the type array is not a compile time constant, add default rank to it.
  using ArgsRank1 = std::conditional_t<!isRank<typename ArgsRank0::template Get<0>>,
                                       typename ArgsRank0::template PushFront<DefaultRank>,
                                       ArgsRank0>;

  // Parse rank from the type array and pop front.
  using TRank = C<ArgsRank1::template Get<0>::value>;
  static_assert(TLayoutsFetched::size == 0 || std::is_same_v<TRank, DefaultRank>,
                "Tensor vector is defined with inconsistent rank");
  using ArgsRankDone = ArgsRank1::template PopFront<>;

  //
  //
  //
  // 2. The second argument is space, either `SpaceHost` or `SpaceDevice`, which is a MUST argument,
  //    so, we require that users should have given this argument.
  //
  // Parse space from the type array and pop front.
  static_assert(ArgsRankDone::size > 0, "Tensor vector space argument not specified");
  using TSpace = ArgsRankDone::template Get<0>;
  static_assert(isSpace<TSpace>, "Tensor vector space should be either `SpaceHost` or `SpaceDevice`");
  using ArgsSpaceDone = ArgsRankDone::template PopFront<>;

  //
  //
  //
  // 3. The third argument is layout, which is optional.
  //
  // Set default layout to dynamic at all dimensions.
  using DefaultLayout = decltype(MakeLayoutMajorWithNZeros<TRank::value>());

  // If the type array is empty, add default layout to it.
  using ArgsLayout0 = std::
      conditional_t<ArgsSpaceDone::size == 0, typename ArgsSpaceDone::template PushFront<DefaultLayout>, ArgsSpaceDone>;

  // If the first element of the type array is not a layout, add default layout to it.
  using ArgsLayout1 = std::conditional_t<!layout::detail::LayoutType<typename ArgsLayout0::template Get<0>>,
                                         typename ArgsLayout0::template PushFront<DefaultLayout>,
                                         ArgsLayout0>;

  // Parse layout from the type array and pop front.
  using TLayout = ArgsLayout1::template Get<0>;
  using ArgsLayoutDone = ArgsLayout1::template PopFront<>;

  //
  //
  //
  // Finalize: Now, the arguments array should be empty.
  static_assert(ArgsLayoutDone::size == 0, "Invalid tensor vector arguments are given");

public:
  // Output a `TypeArray` containing the reduced arguments.
  using Reduced = MakeTypeArray<TRank, TSpace, TLayout>;
};

//
//
//
//
//
//! Now, given any `<T, RawArgs...>`, we are able to generate the "reduced" arguments,
//! which is `<T, TypeArray<...>>`.
//!
//! But there's one question:
//!   using TV0 = TensorVector<T, SpaceHost>;
//!   using TV1 = TensorVector<T, C<1>, SpaceHost>;                            // Explicit specify the rank.
//!   using TV2 = TensorVector<T, std::integral_constant<int, 1>, SpaceHost>;  // Use integral constant.
//!   using TV2 = TensorVector<T, std::integral_constant<uint, 1>, SpaceHost>; // Use uint instead of int.
//!
//! The above four types are differently declared, do they have same type? THEY SHOULD!
//!
//! That is, we should do something to "reduce" the four types to the unique "reduced" type.
//! This is implemented with a magic `using`.
//! After "reduction", their actual types are exactly the same:
//!   using TV0 = TensorVectorReduced<T, C<1>, SpaceHost, Layout<...>>;
//!   using TV1 = TensorVectorReduced<T, C<1>, SpaceHost, Layout<...>>;
//!   using TV2 = TensorVectorReduced<T, C<1>, SpaceHost, Layout<...>>;
//!   using TV3 = TensorVectorReduced<T, C<1>, SpaceHost, Layout<...>>;
//!
//! Now, you are able to understand why we need `class TensorVectorReduced`, and these two lines of code:
//!   template <typename T, typename... RawArgs>
//!   using TensorVector = TensorVectorReduced<T, ...>;
//!
//! We are only going to implement 4 variants of `TensorVectorReduced` with template specialization:
//!   1. Host + static layout.
//!   2. Host + dynamic layout.
//!   3. Device + static layout.
//!   4. Device + dynamic layout.
//! Then, all `TensorVector` declarations can be reduced to these 4 variants.

template <typename T, ArrayType ReducedArgs>
class TensorVectorReduced;

//
//
//
//
//
//! Before rush to implement the 4 variants, you may have noticed that
//! there will be a lot of code duplications.
//! Many functions will have to be copied 4 times.
//!
//! To minimize code duplication, CRTP is used.
//! The base class is `TensorVectorMembers`, whose derived class is `TensorVectorMembers::TDerived`.
//! The 4 derived classes will be implemented later.
//!
//! `TensorVectorMembers` helps generating methods based on some basic methods of `TensorVectorMembers::TDerived`.
//! Here's the methods generation rule:
//!   1. All derived variants should support default construction and construction from a given layout.
//!   2. All derived variants should implement `layout()` and `tensor()`,
//!      which respectively return the underlying layout and a non-owning tensor.
//!   3. `TensorVectorMembers` generate all the remaining methods based on
//!      constructors, `layout()` and `tensor()`.

template <typename T, NonArrayType TRank, NonArrayType TSpace, NonArrayType TLayout>
class TensorVectorMembers {
private:
  // Type of the derived class.
  using TDerived = TensorVectorReduced<T, TypeArray<TRank, TSpace, TLayout>>;

public:
  using value_type = T;

  using Space = TSpace;

  static constexpr auto rank = TLayout::rank;

public:
  // The following `if constexpr`s are necessary because dynamic layouts may also
  // have constant `size`, `size<i>`, and `cosize_safe`.
  [[nodiscard]] constexpr decltype(auto) size() const {
    if constexpr (is_layout_const_size_v<TLayout>)
      return ARIA::size(TLayout{});
    else
      return ARIA::size(layout());
  }

  template <uint i>
  [[nodiscard]] constexpr decltype(auto) size() const {
    if constexpr (is_layout_const_size_at_v<TLayout, i>)
      return ARIA::size<i>(TLayout{});
    else
      return ARIA::size<i>(layout());
  }

  [[nodiscard]] constexpr decltype(auto) cosize_safe() const {
    if constexpr (is_layout_const_cosize_safe_v<TLayout>)
      return ARIA::cosize_safe(TLayout{});
    else
      return ARIA::cosize_safe(layout());
  }

  template <typename... Us>
  [[nodiscard]] constexpr decltype(auto) operator()(Us &&...us) const {
    return tensor()(std::forward<Us>(us)...);
  }

  template <typename... Us>
  [[nodiscard]] constexpr decltype(auto) operator()(Us &&...us) {
    return tensor()(std::forward<Us>(us)...);
  }

  template <typename U>
  [[nodiscard]] constexpr decltype(auto) operator[](U &&u) const {
    return tensor()[std::forward<U>(u)];
  }

  template <typename U>
  [[nodiscard]] constexpr decltype(auto) operator[](U &&u) {
    return tensor()[std::forward<U>(u)];
  }

public:
  // `Realloc` is implemented by simply call the constructor.
  template <typename ULayout>
  constexpr void Realloc(const ULayout &layout) {
    derived() = TDerived(layout);
  }

private:
  // CRTP.
  [[nodiscard]] constexpr const TDerived &derived() const { return static_cast<const TDerived &>(*this); }

  [[nodiscard]] constexpr TDerived &derived() { return static_cast<TDerived &>(*this); }

  // The derived class should implement `layout()` and `tensor()`.
  [[nodiscard]] constexpr decltype(auto) layout() const {
    if constexpr (is_static_v<TLayout>)
      return TLayout{};
    else
      return derived().layout();
  }

  [[nodiscard]] constexpr decltype(auto) tensor() const { return derived().tensor(); }

  [[nodiscard]] constexpr decltype(auto) tensor() { return derived().tensor(); }

private:
  // Implementation of `Mirrored`.

  // Assume `UArray` is a `TypeArray` containing the reduced parameters.

  // Replace its rank with type `V`.
  template <ArrayType UArray, NonArrayType V>
  using RankReplaced = UArray::template Replace<typename UArray::template Get<0>, V>;

  // Replace its space with type `V`.
  template <ArrayType UArray, NonArrayType V>
  using SpaceReplaced = UArray::template Replace<typename UArray::template Get<1>, V>;

  // Try and replace its rank with type `V`, success only when `V` is a valid rank type.
  // If not success, nothing happens.
  template <ArrayType UArray, NonArrayType V>
  using RankTryReplaced = RankReplaced<UArray, std::conditional_t<isRank<V>, V, typename UArray::template Get<0>>>;

  // Try and replace its space with type `V`, success only when `V` is a valid space type.
  // If not success, nothing happens.
  template <ArrayType UArray, NonArrayType V>
  using SpaceTryReplaced = SpaceReplaced<UArray, std::conditional_t<isSpace<V>, V, typename UArray::template Get<1>>>;

  // Try and replace its rank or space with type `V`, success only when `V` is a valid rank or space type.
  // If not success, nothing happens.
  template <ArrayType UArray, NonArrayType V>
  using TryReplaced = SpaceTryReplaced<RankTryReplaced<UArray, V>, V>;

public:
  // Require that the given type `V` is a valid rank or space type.
  // Replace the old rank or space type with the new one.
  // TODO: Mirror of rank cannot be trivially implemented because layout will change, so it is banned for now.
  template <NonArrayType V>
    requires(/*isRank<V> ||*/ isSpace<V>)
  using Mirrored = TensorVectorReduced<T, TryReplaced<TypeArray<TRank, TSpace, TLayout>, V>>;
};

//
//
//
//
//
//! Now, we are ready to implement the 4 variants of `TensorVectorReduced` with template specialization.

// Host + static.
template <NonArrayType T, NonArrayType TStaticLayout>
  requires is_static_v<TStaticLayout>
class TensorVectorReduced<T, TypeArray<cute::C<TStaticLayout::rank>, SpaceHost, TStaticLayout>>
    : public TensorVectorMembers<T, cute::C<TStaticLayout::rank>, SpaceHost, TStaticLayout> {
public:
  using Layout = TStaticLayout;
  using Tensor = cute::Tensor<cute::ArrayEngine<T, cosize_safe_v<TStaticLayout>>, TStaticLayout>;

public:
  TensorVectorReduced() = default;

  explicit TensorVectorReduced(const TStaticLayout &) {}

  ARIA_COPY_MOVE_ABILITY(TensorVectorReduced, default, default);

public:
  [[nodiscard]] constexpr TStaticLayout layout() const { return {}; }

  [[nodiscard]] constexpr decltype(auto) tensor() const { return cute::make_tensor(tensor_.data(), layout()); }

  [[nodiscard]] constexpr decltype(auto) tensor() { return cute::make_tensor(tensor_.data(), layout()); }

  [[nodiscard]] constexpr decltype(auto) rawTensor() const { return tensor(); }

  [[nodiscard]] constexpr decltype(auto) rawTensor() { return tensor(); }

private:
  Tensor tensor_;
};

//
//
//
// Host + dynamic.
template <NonArrayType T, NonArrayType TDynLayout>
  requires(!is_static_v<TDynLayout>)
class TensorVectorReduced<T, TypeArray<cute::C<TDynLayout::rank>, SpaceHost, TDynLayout>>
    : public TensorVectorMembers<T, cute::C<TDynLayout::rank>, SpaceHost, TDynLayout> {
public:
  using Layout = TDynLayout;
  using Tensor = std::decay_t<decltype(cute::make_tensor(std::declval<thrust::host_vector<T>>().data(),
                                                         std::declval<TDynLayout>()))>;

public:
  TensorVectorReduced() = default;

  explicit TensorVectorReduced(const TDynLayout &layout) : engine_(cosize_safe(layout)), layout_(layout) {}

  ARIA_COPY_MOVE_ABILITY(TensorVectorReduced, default, default);

public:
  [[nodiscard]] constexpr decltype(auto) layout() const { return layout_; }

  [[nodiscard]] constexpr decltype(auto) tensor() const { return cute::make_tensor(engine_.data(), layout()); }

  [[nodiscard]] constexpr decltype(auto) tensor() { return cute::make_tensor(engine_.data(), layout()); }

  [[nodiscard]] constexpr decltype(auto) rawTensor() const { return tensor(); }

  [[nodiscard]] constexpr decltype(auto) rawTensor() { return tensor(); }

private:
  thrust::host_vector<T> engine_;
  TDynLayout layout_;
};

//
//
//
// Device + static.
template <NonArrayType T, NonArrayType TStaticLayout>
  requires is_static_v<TStaticLayout>
class TensorVectorReduced<T, TypeArray<cute::C<TStaticLayout::rank>, SpaceDevice, TStaticLayout>>
    : public TensorVectorMembers<T, cute::C<TStaticLayout::rank>, SpaceDevice, TStaticLayout> {
public:
  using Layout = TStaticLayout;
  using Tensor = std::decay_t<decltype(cute::make_tensor(std::declval<thrust::device_vector<T>>().data(),
                                                         std::declval<TStaticLayout>()))>;
  using DeviceTensor = std::decay_t<decltype(cute::make_tensor(std::declval<thrust::device_vector<T>>().data().get(),
                                                               std::declval<TStaticLayout>()))>;

public:
  TensorVectorReduced() = default;

  explicit TensorVectorReduced(const TStaticLayout &) : engine_(cosize_safe_v<TStaticLayout>) {}

  ARIA_COPY_MOVE_ABILITY(TensorVectorReduced, default, default);

public:
  [[nodiscard]] constexpr TStaticLayout layout() const { return {}; }

  [[nodiscard]] constexpr decltype(auto) tensor() const { return cute::make_tensor(engine_.data(), layout()); }

  [[nodiscard]] constexpr decltype(auto) tensor() { return cute::make_tensor(engine_.data(), layout()); }

  [[nodiscard]] constexpr decltype(auto) rawTensor() const {
    return cute::make_tensor(thrust::raw_pointer_cast(engine_.data()), layout());
  }

  [[nodiscard]] constexpr decltype(auto) rawTensor() {
    return cute::make_tensor(thrust::raw_pointer_cast(engine_.data()), layout());
  }

private:
  thrust::device_vector<T> engine_;
};

//
//
//
// Device + dynamic.
template <NonArrayType T, NonArrayType TDynLayout>
  requires(!is_static_v<TDynLayout>)
class TensorVectorReduced<T, TypeArray<cute::C<TDynLayout::rank>, SpaceDevice, TDynLayout>>
    : public TensorVectorMembers<T, cute::C<TDynLayout::rank>, SpaceDevice, TDynLayout> {
public:
  using Layout = TDynLayout;
  using Tensor = std::decay_t<decltype(cute::make_tensor(std::declval<thrust::device_vector<T>>().data(),
                                                         std::declval<TDynLayout>()))>;
  using DeviceTensor = std::decay_t<decltype(cute::make_tensor(std::declval<thrust::device_vector<T>>().data().get(),
                                                               std::declval<TDynLayout>()))>;

public:
  TensorVectorReduced() = default;

  explicit TensorVectorReduced(const TDynLayout &layout) : engine_(cosize_safe(layout)), layout_(layout) {}

  ARIA_COPY_MOVE_ABILITY(TensorVectorReduced, default, default);

public:
  [[nodiscard]] constexpr decltype(auto) layout() const { return layout_; }

  [[nodiscard]] constexpr decltype(auto) tensor() const { return cute::make_tensor(engine_.data(), layout()); }

  [[nodiscard]] constexpr decltype(auto) tensor() { return cute::make_tensor(engine_.data(), layout()); }

  [[nodiscard]] constexpr decltype(auto) rawTensor() const {
    return cute::make_tensor(thrust::raw_pointer_cast(engine_.data()), layout());
  }

  [[nodiscard]] constexpr decltype(auto) rawTensor() {
    return cute::make_tensor(thrust::raw_pointer_cast(engine_.data()), layout());
  }

private:
  thrust::device_vector<T> engine_;
  TDynLayout layout_;
};

//
//
//
//
//
//! As explained above, "reduce" to the 4 variants of `TensorVectorReduced` with `using`.
//! All done!
template <typename T, typename... RawArgs>
using TensorVector = TensorVectorReduced<T, typename TensorVectorArgs<RawArgs...>::Reduced>;

//
//
//
//
//
// `TensorVectorHost` and `TensorVectorDevice` are implemented with a preprocess parser, which
// insert `SpaceHost` or `SpaceDevice` to the proper place of the raw arguments.
template <NonArrayType TSpace, NonArrayType... RawArgs>
// Check: `RawArgs` should not contain spaces.
  requires(isSpace<TSpace> && (!isSpace<RawArgs> && ...))
class CheckAndPreprocessAddSpace {
private:
  // Push the raw args into a `TypeArray`.
  using Args = MakeTypeArray<RawArgs...>;

  // Fetch the user-defined layouts from the arguments, and count the numbers.
  using TLayoutsFetched = Args::template Filter<is_layout>;

  // Push `TSpace` to the back if there's no user-defined layout.
  using ArgsTryPushback =
      std::conditional_t<TLayoutsFetched::size == 0, typename Args::template PushBack<TSpace>, Args>;

  // Insert `TSpace` before the back if there exist user-defined layouts.
  using ArgsTryInsert = std::
      conditional_t<TLayoutsFetched::size >= 1, typename ArgsTryPushback::template Insert<-1, TSpace>, ArgsTryPushback>;

public:
  using Preprocessed = ArgsTryInsert;
};

template <typename T, NonArrayType... RawArgs>
using TensorVectorHost = TensorVector<T, typename CheckAndPreprocessAddSpace<SpaceHost, RawArgs...>::Preprocessed>;

template <typename T, NonArrayType... RawArgs>
using TensorVectorDevice = TensorVector<T, typename CheckAndPreprocessAddSpace<SpaceDevice, RawArgs...>::Preprocessed>;

//
//
//
//
//
template <typename T, NonArrayType... RawArgsNoLayout, layout::detail::LayoutType TLayout>
  requires(!layout::detail::LayoutType<RawArgsNoLayout> && ...)
auto make_tensor_vector(const TLayout &layout) {
  return TensorVector<T, RawArgsNoLayout..., std::decay_t<decltype(layout)>>(layout);
}

//
//
//
//
//
// Whether the given type is a tensor vector.
template <typename T>
struct is_tensor_vector : std::false_type {};

template <typename... Args>
struct is_tensor_vector<TensorVectorReduced<Args...>> : std::true_type {};

template <typename T>
constexpr bool is_tensor_vector_v = is_tensor_vector<T>::value;

template <typename T>
concept TensorVectorType = is_tensor_vector_v<T>;

//
//
//
//
//
// Copy is simply implemented with `thrust::copy`.
template <typename T, ArrayType ArgsDst, ArrayType ArgsSrc>
ARIA_HOST_DEVICE void copy(TensorVectorReduced<T, ArgsDst> &dst, const TensorVectorReduced<T, ArgsSrc> &src) {
  using TVDst = TensorVectorReduced<T, ArgsDst>;
  using TVSrc = TensorVectorReduced<T, ArgsSrc>;

  static_assert(std::is_same_v<typename TVDst::Layout, typename TVSrc::Layout>,
                "Unable to copy tensor vectors with different layouts");

#if ARIA_IS_HOST_CODE
  if (dst.size() != src.size())
    ARIA_THROW(std::runtime_error, "Unable to copy tensor vectors with different sizes");
  if (dst.cosize_safe() != src.cosize_safe())
    ARIA_THROW(std::runtime_error, "Unable to copy tensor vectors with different cosizes");
#else
  ARIA_ASSERT(dst.size() == src.size(), "Unable to copy tensor vectors with different sizes");
  ARIA_ASSERT(dst.cosize_safe() == src.cosize_safe(), "Unable to copy tensor vectors with different cosizes");
#endif

  thrust::copy(src.tensor().data(), src.tensor().data() + src.cosize_safe(), dst.tensor().data());
}

} // namespace tensor_vector::detail

} // namespace ARIA
