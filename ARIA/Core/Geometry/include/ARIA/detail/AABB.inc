#pragma once

namespace ARIA {

namespace aabb::detail {

// See the comments below.
template <typename T, auto d, auto n, typename U, typename... Us>
static decltype(auto) ConstructVecWithNEqualedValuesImpl(const U &v, Us &&...vs) {
  if constexpr (n == 0)
    return Vec<T, d>(std::forward<Us>(vs)...);
  else
    return ConstructVecWithNEqualedValuesImpl<T, d, n - 1>(v, v, std::forward<Us>(vs)...);
}

// A function wrapper which calls the constructor of `Vec<T, d>` with `d` equaled values.
template <typename T, auto d>
static decltype(auto) ConstructVecWithNEqualedValues(const T &value) {
  return ConstructVecWithNEqualedValuesImpl<T, d, d>(value);
}

} // namespace aabb::detail

//
//
//
//
//
template <typename T, auto d>
ARIA_HOST_DEVICE inline /*constexpr*/ AABB<T, d>::AABB()
    : infAndSup_{aabb::detail::ConstructVecWithNEqualedValues<T, d>(supremum<T>),
                 aabb::detail::ConstructVecWithNEqualedValues<T, d>(infimum<T>)} {}

//
//
//
//
//
template <typename T, auto d>
ARIA_HOST_DEVICE inline /*constexpr*/ const Vec<T, d> &AABB<T, d>::operator[](uint i) const {
  return infAndSup_[i];
}

template <typename T, auto d>
ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> &AABB<T, d>::operator[](uint i) {
  return infAndSup_[i];
}

//
//
//
//
//
template <typename T, auto d>
ARIA_HOST_DEVICE inline /*constexpr*/ bool AABB<T, d>::empty() const {
  bool res = false;

  ForEach<d>([&]<auto i>() {
    if (res)
      return;

    if (inf()[i] > sup()[i])
      res = true;
  });

  return res;
}

template <typename T, auto d>
template <typename... Args>
ARIA_HOST_DEVICE inline /*constexpr*/ void AABB<T, d>::Unionize(Args &&...args) {
  *this = unionized(*this, std::forward<Args>(args)...);
}

//
//
//
//
//
template <typename T, auto d>
ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> AABB<T, d>::diagonal() const {
  return sup() - inf();
}

template <typename T, auto d>
ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> AABB<T, d>::center() const {
  return (sup() + inf()) / T{2};
}

template <typename T, auto d>
ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, d> AABB<T, d>::offset(const Vec<T, d> &p) const {
  return (p - inf()).cwiseQuotient(sup() - inf());
}

//
//
//
//
//
template <typename T, auto d>
template <typename... Args>
ARIA_HOST_DEVICE inline /*constexpr*/ AABB<T, d> AABB<T, d>::unionized(Args &&...args) {
  // Union an `AABB` with another `AABB` or a `Vec`.
  auto unionizedOne = Overload{[](const AABB &a, const AABB &b) {
    AABB res;
    ForEach<d>([&]<auto i>() { res.inf()[i] = std::min(a.inf()[i], b.inf()[i]); });
    ForEach<d>([&]<auto i>() { res.sup()[i] = std::max(a.sup()[i], b.sup()[i]); });
    return res;
  }, [](const AABB &a, const Vec<T, d> &b) {
    AABB res;
    ForEach<d>([&]<auto i>() { res.inf()[i] = std::min(a.inf()[i], b[i]); });
    ForEach<d>([&]<auto i>() { res.sup()[i] = std::max(a.sup()[i], b[i]); });
    return res;
  }};

  //! Then, we want to call `unionizedOne` with `args`.
  //!
  //! Here, it is important to optimize for the first argument.
  //! 1. If the first argument is an `AABB`,
  //!    directly set `res` to it, and call `unionizedOne` with the remaining arguments.
  //! 2. If the first argument is a `Vec`,
  //!    directly set `inf` and `sup` of `res` to the `Vec`, and call `unionizedOne` with the remaining arguments.
  //!
  //! This optimization is important, because for example,
  //! if this method is called with one `AABB` or one `Vec`,
  //! no computation will be performed.
  auto unionizedImpl = Overload{[&]<typename... Ts>(const AABB &t0, Ts &&...ts) { // If the first argument is an `AABB`.
    // Directly set `res` to it.
    AABB res = t0;

    // Call `unionizedOne` with the remaining arguments.
    auto unionizeOne = [&](const auto &b) { res = unionizedOne(res, b); };
    (unionizeOne(ts), ...);

    return res;
  }, [&]<typename... Ts>(const Vec<T, d> &t0, Ts &&...ts) { // If the first argument is a `Vec`.
    // Directly set `inf` and `sup` of `res` to the `Vec`.
    AABB res;
    res.inf() = t0;
    res.sup() = t0;

    // Call `unionizedOne` with the remaining arguments.
    auto unionizeOne = [&](const auto &b) { res = unionizedOne(res, b); };
    (unionizeOne(ts), ...);

    return res;
  }};

  // Finally, we call the overloaded `unionizedImpl` with `args`.
  return unionizedImpl(std::forward<Args>(args)...);
}

//
//
//
//
//
//
//
//
//
namespace aabb::detail {

// Whether the given type is `AABB<T, ...>`.
template <typename T>
struct is_aabb : std::false_type {};

template <typename T, auto d>
struct is_aabb<AABB<T, d>> : std::true_type {};

template <typename T>
static constexpr bool is_aabb_v = is_aabb<T>::value;

//
//
//
// Whether the given type is `AABB<T, d>`.
template <typename T, auto d>
struct is_aabb_d : std::false_type {};

template <typename T, auto d>
struct is_aabb_d<AABB<T, d>, d> : std::true_type {};

template <typename T, auto d>
static constexpr bool is_aabb_d_v = is_aabb_d<T, d>::value;

} // namespace aabb::detail

} // namespace ARIA
