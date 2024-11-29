#include "ARIA/Let.h"
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

TEST(Let, Base) {
  Transform t;

  {
    // let v0 = t.forward();
    let v1 = Let(t.forward());
    let &v2 = v1;
    let &&v3 = std::move(v1);
    static_assert(std::is_same_v<decltype(v1), Vec3<float>>);
    static_assert(std::is_same_v<decltype(v2), Vec3<float> &>);
    static_assert(std::is_same_v<decltype(v3), Vec3<float> &&>);
  }

  {
    // let v0 = t.forward().x();
    let v1 = Let(t.forward().x());
    let &v2 = v1;
    let &&v3 = std::move(v1);
    static_assert(std::is_same_v<decltype(v1), float>);
    static_assert(std::is_same_v<decltype(v2), float &>);
    static_assert(std::is_same_v<decltype(v3), float &&>);
  }

  {
    std::vector<bool> vec(10);
    // let v0 = vec[0];
    let v1 = Let(vec[0]);
    let &v2 = v1;
    let &&v3 = std::move(v1);
    static_assert(std::is_same_v<decltype(v1), bool>);
    static_assert(std::is_same_v<decltype(v2), bool &>);
    static_assert(std::is_same_v<decltype(v3), bool &&>);
  }
}

TEST(Let, Thrust) {
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
    float storage = 0;
    thrust::device_reference<float> storageRef{thrust::device_reference<float>::pointer{&storage}};
    // let v0 = storageRef;
    let v1 = Let(storageRef);
    static_assert(std::is_same_v<decltype(v1), float>);
  }

  {
    thrust::device_vector<float> storage;
    static_assert(std::is_same_v<decltype(Let(storage[0])), float>);
  }
}

} // namespace ARIA
