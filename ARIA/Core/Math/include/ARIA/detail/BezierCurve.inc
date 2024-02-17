#pragma once

namespace ARIA {

// Specialization for compile-time degrees.
template <typename T, auto dim, uint degree, typename TControlPoints>
class BezierCurve<T, dim, Degree<degree>, TControlPoints> {
public:
  using value_type = T;

private:
  using VecDim = Vec<T, dim>;
  static_assert(std::is_same_v<VecDim, typename TControlPoints::value_type>,
                "Type of control points does not match the dimension");
  //! Should not use `std::decay_t<decltype(std::declval<TControlPoints>()[0])>` in order to support proxy systems.

public:
  BezierCurve() = default;

  ARIA_HOST_DEVICE explicit BezierCurve(const TControlPoints &controlPoints) : controlPoints_(controlPoints) {}

  ARIA_COPY_MOVE_ABILITY(BezierCurve, default, default); //! Let `TControlPoints` decide.

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, controlPoints, controlPoints_);

public:
  [[nodiscard]] ARIA_HOST_DEVICE constexpr bool IsInDomain(const T &t) const { return T{0} <= t && t <= T{1}; }

  [[nodiscard]] ARIA_HOST_DEVICE VecDim operator()(const T &t) const {
    ARIA_ASSERT(IsInDomain(t));

    // Apply the deCasteljau algorithm, 1997, The NURBS Book, 24.
    std::array<VecDim, nCPs> temp;

    ForEach<nCPs>([&]<auto i>() { temp[i] = controlPoints()[i]; });
    ForEach<degree>([&]<auto _r>() {
      constexpr auto round = _r + 1;
      ForEach<nCPs - round>([&]<auto i>() { temp[i] = Lerp(temp[i], temp[i + 1], t); });
    });

    return temp[0];
  }

private:
  // `CP` is an abbreviation of "control point".
  static constexpr uint nCPs = degree + 1;

  TControlPoints controlPoints_;
};

//
//
//
// Specialization for runtime degrees.
template <typename T, auto dim, typename TControlPoints>
class BezierCurve<T, dim, DegreeDynamic, TControlPoints> {
public:
  using value_type = T;

private:
  using VecDim = Vec<T, dim>;
  static_assert(std::is_same_v<VecDim, typename TControlPoints::value_type>,
                "Type of control points does not match the dimension");
  //! Should not use `std::decay_t<decltype(std::declval<TControlPoints>()[0])>` in order to support proxy systems.

public:
  BezierCurve() = default;

  ARIA_HOST_DEVICE explicit BezierCurve(const TControlPoints &controlPoints) : controlPoints_(controlPoints) {}

  ARIA_COPY_MOVE_ABILITY(BezierCurve, default, default); //! Let `TControlPoints` decide.

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, controlPoints, controlPoints_);

public:
  [[nodiscard]] ARIA_HOST_DEVICE constexpr bool IsInDomain(const T &t) const { return T{0} <= t && t <= T{1}; }

  [[nodiscard]] ARIA_HOST_DEVICE VecDim operator()(const T &t) const {
    ARIA_ASSERT(IsInDomain(t));

    // `CP` is an abbreviation of "control point".
    const uint nCPs = controlPoints().size();

#if ARIA_IS_HOST_CODE
    // Apply the deCasteljau algorithm, 1997, The NURBS Book, 24.
    small_vector<VecDim, 8> temp(nCPs); //! Local buffer optimization for degrees <= 7.

    for (uint i = 0; i < nCPs; ++i)
      temp[i] = controlPoints()[i];
    for (uint round = 1; round < nCPs; ++round)
      for (uint i = 0; i < nCPs - round; ++i)
        temp[i] = Lerp(temp[i], temp[i + 1], t);

    return temp[0];
#else
    // Compute the combinations.
    // See https://stackoverflow.com/questions/11809502/which-is-better-way-to-calculate-ncr.
    auto nCr = [](uint n, uint r) {
      if (r > n - r)
        r = n - r; // Because C(n, r) == C(n, n - r).

      uint res = 1;
      for (uint i = 1; i <= r; i++) {
        res *= n - r + i;
        res /= i;
      }

      return res;
    };

    const uint degree = nCPs - 1;
    VecDim pos = VecCP::Zero();

    // Directly compute the Bernstein polynomials.
    for (uint i = 0; i < nCPs; ++i) {
      VecDim cp = controlPoints()[i];
      pos += cp * nCr(degree, i) * pow(1 - t, degree - i) * pow(t, i);
    }

    return pos;
#endif
  }

private:
  TControlPoints controlPoints_;
};

} // namespace ARIA
