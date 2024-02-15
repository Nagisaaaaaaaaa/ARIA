#pragma once

#include "ARIA/ForEach.h"
#include "ARIA/Vec.h"

namespace ARIA {

/// \note `AABB` is implemented with `Vec`, which internally uses `Eigen::Vector`.
/// So, all functions are currently not `constexpr`.
template <typename T, auto s>
class AABB final {
public:
  // TODO: Define infinity, max, min, and suprime at Math.h.
  ARIA_HOST_DEVICE inline /*constexpr*/ AABB()
      : pMin_(ConstructVecWithNEqualedValues(10000)), pMax_(ConstructVecWithNEqualedValues(-10000)) {}

  template <typename... Args>
    requires(sizeof...(Args) > 0 &&  // Size `> 0` to avoid conflict with the default constructor.
             (sizeof...(Args) > 1 || // If size `> 1`, safe. If size `== 1`, may conflict with the copy constructor.
              (!std::is_same_v<std::decay_t<Args>, AABB> && ...))) // So, requires that the argument type is not AABB.
  ARIA_HOST_DEVICE inline /*constexpr*/ explicit AABB(Args &&...args) : AABB() {
    Unionize(std::forward<Args>(args)...);
  }

  ARIA_COPY_MOVE_ABILITY(AABB, default, default);
  ~AABB() = default;

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, pMin, pMin_);
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, pMax, pMax_);

public:
  ARIA_HOST_DEVICE inline /*constexpr*/ bool empty() const {
    bool res = false;

    ForEach<s>([&]<auto i>() {
      if (res)
        return;

      if (pMin()[i] > pMax()[i])
        res = true;
    });

    return res;
  }

  template <typename... Args>
  ARIA_HOST_DEVICE static inline /*constexpr*/ AABB unionized(Args &&...args) {
    // Union an `AABB` with another `AABB` or a `Vec`.
    auto unionizedOne = Overload{[](const AABB &a, const AABB &b) {
      AABB res;
      ForEach<s>([&]<auto i>() { res.pMin()[i] = std::min(a.pMin()[i], b.pMin()[i]); });
      ForEach<s>([&]<auto i>() { res.pMax()[i] = std::max(a.pMax()[i], b.pMax()[i]); });
      return res;
    }, [](const AABB &a, const Vec<T, s> &b) {
      AABB res;
      ForEach<s>([&]<auto i>() { res.pMin()[i] = std::min(a.pMin()[i], b[i]); });
      ForEach<s>([&]<auto i>() { res.pMax()[i] = std::max(a.pMax()[i], b[i]); });
      return res;
    }};

    //! Then, we want to call `unionizedOne` with `args`.
    //!
    //! Here, it is important to optimize for the first argument.
    //! 1. If the first argument is an `AABB`,
    //!    directly set it to `res`, and call `unionizedOne` with the remaining arguments.
    //! 2. If the first argument is a `Vec`,
    //!    directly set `pMin` and `pMax` of `res` to the `Vec`, and call `unionizedOne` with the remaining arguments.
    //!
    //! This optimization is important, because for example,
    //! if this method is called with one `AABB` or one `Vec`,
    //! no computation will be performed.
    auto unionizedImpl =
        Overload{[&]<typename... Ts>(const AABB &t0, Ts &&...ts) { // If the first argument is an `AABB`.
      AABB res = t0;

      auto unionizeOne = [&](const auto &b) { res = unionizedOne(res, b); };
      (unionizeOne(ts), ...);

      return res;
    }, [&]<typename... Ts>(const Vec<T, s> &t0, Ts &&...ts) { // If the first argument is a `Vec`.
      AABB res;
      res.pMin() = t0;
      res.pMax() = t0;

      auto unionizeOne = [&](const auto &b) { res = unionizedOne(res, b); };
      (unionizeOne(ts), ...);

      return res;
    }};

    // Finally, we call the overloaded `unionizedImpl` with `args`.
    return unionizedImpl(std::forward<Args>(args)...);
  }

  template <typename... Args>
  ARIA_HOST_DEVICE inline /*constexpr*/ void Unionize(Args &&...args) {
    *this = unionized(*this, std::forward<Args>(args)...);
  }

  ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, s> center() const { return (pMax() + pMin()) / T{2}; }

  ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, s> offset(const Vec<T, s> &p) const {
    Vec<T, s> o;
    ForEach<s>([&]<auto i>() { o[i] = (p[i] - pMin()[i]) / (pMax()[i] - pMin()[i]); });
    return o;
  }

  ARIA_HOST_DEVICE inline /*constexpr*/ Vec<T, s> diagonal() const { return pMax() - pMin(); }

private:
  Vec<T, s> pMin_;
  Vec<T, s> pMax_;

  // A function wrapper which calls the constructor of `Vec<T, s>` with `s` equaled values.
  static decltype(auto) ConstructVecWithNEqualedValues(const T &value) {
    return ConstructVecWithNEqualedValuesImpl<s>(value);
  }

  template <uint n, typename U, typename... Us>
  static decltype(auto) ConstructVecWithNEqualedValuesImpl(const U &v, Us &&...vs) {
    if constexpr (n == 0)
      return Vec<T, s>(std::forward<Us>(vs)...);
    else
      return ConstructVecWithNEqualedValuesImpl<n - 1>(v, v, std::forward<Us>(vs)...);
  }
};

//
//
//
template <typename T>
using AABB1 = AABB<T, 1>;
template <typename T>
using AABB2 = AABB<T, 2>;
template <typename T>
using AABB3 = AABB<T, 3>;
template <typename T>
using AABB4 = AABB<T, 4>;

using AABB1i = AABB1<int>;
using AABB1u = AABB1<uint>;
using AABB1f = AABB1<float>;
using AABB1d = AABB1<double>;
using AABB1r = AABB1<Real>;

using AABB2i = AABB2<int>;
using AABB2u = AABB2<uint>;
using AABB2f = AABB2<float>;
using AABB2d = AABB2<double>;
using AABB2r = AABB2<Real>;

using AABB3i = AABB3<int>;
using AABB3u = AABB3<uint>;
using AABB3f = AABB3<float>;
using AABB3d = AABB3<double>;
using AABB3r = AABB3<Real>;

using AABB4i = AABB4<int>;
using AABB4u = AABB4<uint>;
using AABB4f = AABB4<float>;
using AABB4d = AABB4<double>;
using AABB4r = AABB4<Real>;

} // namespace ARIA
