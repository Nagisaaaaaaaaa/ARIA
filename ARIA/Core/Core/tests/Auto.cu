#include "ARIA/Auto.h"
#include "ARIA/Property.h"

#include <gtest/gtest.h>

#include <thrust/device_vector.h>

#include <cmath>

namespace ARIA {

namespace {

template <typename T>
class Vec3 {
public:
  Vec3() = default;

  Vec3(const T &x, const T &y, const T &z) : x_(x), y_(y), z_(z) {}

  ARIA_REF_PROP(public, , x, x_);
  ARIA_REF_PROP(public, , y, y_);
  ARIA_REF_PROP(public, , z, z_);

  ARIA_PROP(public, private, , T, lengthSqr);
  ARIA_PROP(public, public, , T, length);

private:
  T x_{}, y_{}, z_{};

  [[nodiscard]] auto ARIA_PROP_GETTER(lengthSqr)() const { return x() * x() + y() * y() + z() * z(); }

  [[nodiscard]] auto ARIA_PROP_GETTER(length)() const { return std::sqrt(lengthSqr()); }

  void ARIA_PROP_SETTER(length)(const auto &value) {
    auto scaling = value / length();
    x() *= scaling;
    y() *= scaling;
    z() *= scaling;
  }
};

class Transform {
public:
  ARIA_PROP_BEGIN(public, public, , Vec3<float>, forward);
  ARIA_SUB_PROP(, float, x);
  ARIA_SUB_PROP(, float, y);
  ARIA_SUB_PROP(, float, z);
  ARIA_SUB_PROP(, float, lengthSqr);
  ARIA_SUB_PROP(, float, length);
  ARIA_PROP_END;

private:
  Vec3<double> forward_;

  [[nodiscard]] auto ARIA_PROP_GETTER(forward)() const {
    return Vec3<float>{float(forward_.x()), float(forward_.y()), float(forward_.z())};
  }

  void ARIA_PROP_SETTER(forward)(const auto &value) {
    forward_.x() = value.x();
    forward_.y() = value.y();
    forward_.z() = value.z();
  }
};

} // namespace

TEST(Auto, Base) {
  Transform t;

  {
    auto v0 = t.forward();
    auto v1 = Auto(t.forward());
    static_assert(!std::is_same_v<decltype(v0), Vec3<float>>);
    static_assert(std::is_same_v<decltype(v1), Vec3<float>>);
  }

  {
    auto v0 = t.forward().x();
    auto v1 = Auto(t.forward().x());
    static_assert(!std::is_same_v<decltype(v0), float>);
    static_assert(std::is_same_v<decltype(v1), float>);
  }

  {
    std::vector<bool> vec(10);
    auto v0 = vec[0];
    auto v1 = Auto(vec[0]);
    static_assert(!std::is_same_v<decltype(v0), bool>);
    static_assert(std::is_same_v<decltype(v1), bool>);
  }
}

TEST(Auto, Thrust) {
  {
    float v = 0;
    static_assert(!std::is_same_v<decltype(v), decltype(thrust::raw_reference_cast(v))>);
    static_assert(std::is_same_v<std::add_lvalue_reference_t<decltype(v)>, decltype(thrust::raw_reference_cast(v))>);
  }

  {
    float storage = 0;
    float &v = storage;
    static_assert(std::is_same_v<decltype(v), decltype(thrust::raw_reference_cast(v))>);
  }

  {
    float &&v = 0;
    static_assert(!std::is_same_v<decltype(v), decltype(thrust::raw_reference_cast(v))>);
    static_assert(std::is_same_v<std::add_lvalue_reference_t<std::remove_reference_t<decltype(v)>>,
                                 decltype(thrust::raw_reference_cast(v))>);
  }

  {
    thrust::device_vector<float> storage{3.14F};
    static_assert(std::is_same_v<decltype(Auto(storage[0])), float>);
    EXPECT_FLOAT_EQ(Auto(storage[0]), 3.14F);
  }
}

} // namespace ARIA
