#include "ARIA/Property.h"
#include "ARIA/Constant.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace ARIA {

//! Forbid clang-format for this file because this file is extremely weird.
// clang-format off

#if 0
#include <cxxabi.h>

template <typename T>
[[nodiscard]] static char* GetTypeName() {
  int status = 0;
  char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
  return demangled;
}
#endif



namespace {

class A {
 public:
  ARIA_PROP_BEGIN(public, public, , int, x);
  ARIA_PROP_END;

 private:
  int x_{};

  [[nodiscard]] int ARIA_PROP_GETTER(x)() const {
    return x_;
  }

  void ARIA_PROP_SETTER(x)(const int& value) {
    x_ = value;
  }

  friend std::ostream& operator<<(std::ostream& os, const A& a){
    os << a.x().value();
    return os;
  }
};



class B {
 public:
  ARIA_PROP_BEGIN(public, public, , int, x);
  ARIA_PROP_END;

 private:
  int x_{};

  [[nodiscard]] int ARIA_PROP_GETTER(x)() const {
    return x_;
  }

  void ARIA_PROP_SETTER(x)(const int& value) {
    x_ = value;
  }
};



// All return reference
struct HierarchyA {
  struct X {
    struct Y {
      int z_ = 5;
      [[nodiscard]] const auto& z() const {
        return z_;
      }
      [[nodiscard]] auto& z() {
        return z_;
      }
    } y_;
    [[nodiscard]] const auto& y() const {
      return y_;
    }
    [[nodiscard]] auto& y() {
      return y_;
    }
  } x_;
  [[nodiscard]] const auto& x() const {
    return x_;
  }
  [[nodiscard]] auto& x() {
    return x_;
  }



  struct X1 {
    int y0_ = 10;
    [[nodiscard]] const auto& y0() const {
      return y0_;
    }
    [[nodiscard]] auto& y0() {
      return y0_;
    }

    float y1_ = 20;
    [[nodiscard]] const auto& y1() const {
      return y1_;
    }
    [[nodiscard]] auto& y1() {
      return y1_;
    }
  } x1_;
  [[nodiscard]] const auto& x1() const {
    return x1_;
  }
  [[nodiscard]] auto& x1() {
    return x1_;
  }
};



// All return copy
struct HierarchyB {
  struct X {
    struct Y {
      int z_ = 5;
      [[nodiscard]] auto z() const {
        return z_;
      }
    } y_;
    [[nodiscard]] auto y() const {
      return y_;
    }
  } x_;
  [[nodiscard]] auto x() const {
    return x_;
  }



  struct X1 {
    int y0_ = 10;
    [[nodiscard]] auto y0() const {
      return y0_;
    }

    float y1_ = 20;
    [[nodiscard]] auto y1() const {
      return y1_;
    }
  } x1_;
  [[nodiscard]] auto x1() const {
    return x1_;
  }
};



// Only the last 2 property returns copy
struct HierarchyC {
  struct X {
    struct Y {
      int z_ = 5;
      [[nodiscard]] auto z() const {
        return z_;
      }
    } y_;
    [[nodiscard]] auto y() const {
      return y_;
    }
  } x_;
  [[nodiscard]] const auto& x() const {
    return x_;
  }
  [[nodiscard]] auto& x() {
    return x_;
  }



  struct X1 {
    int y0_ = 10;
    [[nodiscard]] const auto& y0() const {
      return y0_;
    }
    [[nodiscard]] auto& y0() {
      return y0_;
    }

    float y1_ = 20;
    [[nodiscard]] const auto& y1() const {
      return y1_;
    }
    [[nodiscard]] auto& y1() {
      return y1_;
    }
  } x1_;
  [[nodiscard]] const auto& x1() const {
    return x1_;
  }
  [[nodiscard]] auto& x1() {
    return x1_;
  }
};



// All return reference
class ChaosA {
 public:
  ARIA_PROP_BEGIN(public, public, , HierarchyA, h);
    ARIA_SUB_PROP_BEGIN(, HierarchyA::X, x);
      ARIA_SUB_PROP_BEGIN(, HierarchyA::X::Y, y);
        ARIA_SUB_PROP_BEGIN(, int, z);
        ARIA_SUB_PROP_END;
      ARIA_SUB_PROP_END;
    ARIA_SUB_PROP_END;

    ARIA_SUB_PROP_BEGIN(, HierarchyA::X1, x1);
      ARIA_SUB_PROP_BEGIN(, int, y0);
      ARIA_SUB_PROP_END;
      ARIA_SUB_PROP_BEGIN(, float, y1);
      ARIA_SUB_PROP_END;
    ARIA_SUB_PROP_END;
  ARIA_PROP_END;

 public:
  HierarchyA h_{};

  [[nodiscard]] auto ARIA_PROP_GETTER(h)() const {
    return h_;
  }

  void ARIA_PROP_SETTER(h)(const auto& value) {
    h_ = value;
  }
};



// All return copy
class ChaosB {
 public:
  ARIA_PROP_BEGIN(public, public, , HierarchyB, h);
    ARIA_SUB_PROP_BEGIN(, HierarchyB::X, x);
      ARIA_SUB_PROP_BEGIN(, HierarchyB::X::Y, y);
        ARIA_SUB_PROP_BEGIN(, int, z);
        ARIA_SUB_PROP_END;
      ARIA_SUB_PROP_END;
    ARIA_SUB_PROP_END;

    ARIA_SUB_PROP_BEGIN(, HierarchyB::X1, x1);
      ARIA_SUB_PROP_BEGIN(, int, y0);
      ARIA_SUB_PROP_END;
      ARIA_SUB_PROP_BEGIN(, float, y1);
      ARIA_SUB_PROP_END;
    ARIA_SUB_PROP_END;
  ARIA_PROP_END;

 public:
  HierarchyB h_{};

  [[nodiscard]] auto ARIA_PROP_GETTER(h)() const {
    return h_;
  }

  void ARIA_PROP_SETTER(h)(const auto& value) {
    h_ = value;
  }
};



// Only the last 2 property returns copy
class ChaosC {
 public:
  ARIA_PROP_BEGIN(public, public, , HierarchyC, h);
    ARIA_SUB_PROP_BEGIN(, HierarchyC::X, x);
      ARIA_SUB_PROP_BEGIN(, HierarchyC::X::Y, y);
        ARIA_SUB_PROP_BEGIN(, int, z);
        ARIA_SUB_PROP_END;
      ARIA_SUB_PROP_END;
    ARIA_SUB_PROP_END;

    ARIA_SUB_PROP_BEGIN(, HierarchyC::X1, x1);
      ARIA_SUB_PROP_BEGIN(, int, y0);
      ARIA_SUB_PROP_END;
      ARIA_SUB_PROP_BEGIN(, float, y1);
      ARIA_SUB_PROP_END;
    ARIA_SUB_PROP_END;
  ARIA_PROP_END;

 public:
  HierarchyC h_{};

  [[nodiscard]] auto ARIA_PROP_GETTER(h)() const {
    return h_;
  }

  void ARIA_PROP_SETTER(h)(const auto& value) {
    h_ = value;
  }
};



// Long dependency
class ChaosListA {
 public:
  struct A {
    struct B {
      struct C {
        struct D {
          struct E {
            struct F {
              ARIA_PROP_BEGIN(public, public, , int, g);
              ARIA_SUB_PROP_END;

             public:
              int g_ = 5;
              [[nodiscard]] auto ARIA_PROP_GETTER(g)() const {
                return g_;
              }
              void ARIA_PROP_SETTER(g)(const auto& value) {
                g_ = value;
              }
            } f_;

            [[nodiscard]] auto f() const {
              return f_;
            }
            void f(const auto& value) {
              f_ = value;
            }
          } e_;

          [[nodiscard]] const auto& e() const {
            return e_;
          }
          [[nodiscard]] auto& e() {
            return e_;
          }
        } d_;

        ARIA_PROP_BEGIN(public, public, , D, d);
          ARIA_SUB_PROP_BEGIN(, D::E, e);
            ARIA_SUB_PROP_BEGIN(, D::E::F, f);
              ARIA_SUB_PROP_BEGIN(, int, g);
              ARIA_SUB_PROP_END;
            ARIA_SUB_PROP_END;
          ARIA_SUB_PROP_END;
        ARIA_PROP_END;

        [[nodiscard]] auto ARIA_PROP_GETTER(d)() const {
          return d_;
        }
        void ARIA_PROP_SETTER(d)(const auto& value) {
          d_ = value;
        }
      } c_;

      [[nodiscard]] auto c() const {
        return c_;
      }
      void c(const auto& value) {
        c_ = value;
      }
    } b_;

    [[nodiscard]] const auto& b() const {
      return b_;
    }
    [[nodiscard]] auto& b() {
      return b_;
    }
  } a_;

  ARIA_PROP_BEGIN(public, public, , A, a);
    ARIA_SUB_PROP_BEGIN(, A::B, b);
      ARIA_SUB_PROP_BEGIN(, A::B::C, c);
        ARIA_SUB_PROP_BEGIN(, A::B::C::D, d);
          ARIA_SUB_PROP_BEGIN(, A::B::C::D::E, e);
            ARIA_SUB_PROP_BEGIN(, A::B::C::D::E::F, f);
              ARIA_SUB_PROP_BEGIN(, int, g);
              ARIA_SUB_PROP_END;
            ARIA_SUB_PROP_END;
          ARIA_SUB_PROP_END;
        ARIA_SUB_PROP_END;
      ARIA_SUB_PROP_END;
    ARIA_SUB_PROP_END;
  ARIA_PROP_END;

  [[nodiscard]] auto ARIA_PROP_GETTER(a)() const {
    return a_;
  }
  void ARIA_PROP_SETTER(a)(const auto& value) {
    a_ = value;
  }
};



// Long long dependency
class ChaosListB {
 public:
  ARIA_PROP_BEGIN(public, public, , ChaosListA, a);
    ARIA_SUB_PROP_BEGIN(, ChaosListA::A, a);
      ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B, b);
        ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B::C, c);
          ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B::C::D, d);
            ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B::C::D::E, e);
              ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B::C::D::E::F, f);
                ARIA_SUB_PROP_BEGIN(, int, g); // Compile error if set to other types
                ARIA_SUB_PROP_END;
              ARIA_SUB_PROP_END;
            ARIA_SUB_PROP_END;
          ARIA_SUB_PROP_END;
        ARIA_SUB_PROP_END;
      ARIA_SUB_PROP_END;
    ARIA_PROP_END;
  ARIA_PROP_END;

 public:
  ChaosListA a_{};

  [[nodiscard]] auto ARIA_PROP_GETTER(a)() const {
    return a_;
  }

  void ARIA_PROP_SETTER(a)(const auto& value) {
    a_ = value;
  }
};



// Long long long dependency
class ChaosListC {
 public:
  ARIA_PROP_BEGIN(public, public, , ChaosListB, b);
    ARIA_SUB_PROP_BEGIN(, ChaosListA, a);
      ARIA_SUB_PROP_BEGIN(, ChaosListA::A, a);
        ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B, b);
          ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B::C, c);
            ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B::C::D, d);
              ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B::C::D::E, e);
                ARIA_SUB_PROP_BEGIN(, ChaosListA::A::B::C::D::E::F, f);
                  ARIA_SUB_PROP_BEGIN(, int, g); // Compile error if set to other types
                  ARIA_SUB_PROP_END;
                ARIA_SUB_PROP_END;
              ARIA_SUB_PROP_END;
            ARIA_SUB_PROP_END;
          ARIA_SUB_PROP_END;
        ARIA_SUB_PROP_END;
      ARIA_PROP_END;
    ARIA_PROP_END;
  ARIA_PROP_END;

 public:
  ChaosListB b_{};

  [[nodiscard]] auto ARIA_PROP_GETTER(b)() const {
    return b_;
  }

  void ARIA_PROP_SETTER(b)(const auto& value) {
    b_ = value;
  }
};



// Cooperation with other proxies
class VectorBool {
 public:
  ARIA_PROP(public, private, , std::vector<bool>, vs);

  ARIA_PROP(public, public, , bool, v);

 private:
  std::vector<bool> vs_{false};

  [[nodiscard]] auto ARIA_PROP_GETTER(vs)() const {
    return vs_;
  }

  [[nodiscard]] auto ARIA_PROP_GETTER(v)() const {
    return vs_[0];
  }
  void ARIA_PROP_SETTER(v)(const auto& value) {
    vs_[0] = value;
  }
};



// Callable
class Func {
 public:
  ARIA_PROP(public, private, , std::function<int(int)>, f0);

  ARIA_PROP(public, private, , std::function<void(int)>, f1);

 private:
  int offset = 1;

  std::function<int(int)> f0_ = [=] (int v) {
    std::stringstream ss;
    ss << "Testing callable: " << offset + v << std::endl;
    return offset + v;
  };

  std::function<void(int)> f1_ = [=] (int v) {
    std::stringstream ss;
    ss << "Testing callable: " << offset + v << std::endl;
  };

  [[nodiscard]] auto ARIA_PROP_GETTER(f0)() const {
    return f0_;
  }

  [[nodiscard]] auto ARIA_PROP_GETTER(f1)() const {
    return f1_;
  }
};



// Reference properties
template <typename T>
class Vec3 {
 public:
  Vec3() = default;
  Vec3(const T& x, const T& y, const T& z) : x_(x), y_(y), z_(z) {}

  ARIA_REF_PROP(public, , x, x_);
  ARIA_REF_PROP(public, , y, y_);
  ARIA_REF_PROP(public, , z, z_);

  ARIA_REF_PROP(public, , xx, xImpl());
  ARIA_REF_PROP(public, , yy, yImpl());
  ARIA_REF_PROP(public, , zz, [this] () -> T& { return this->z_; }());

 public:
  T Func0(const T& inc) {
    return x();
  }
  void Func1(const T& inc) {
  }
  T Func2(const T& inc) const {
    return x();
  }
  void Func3(const T& inc) const {
  }

 public:
  friend std::ostream& operator<<(std::ostream& os, const Vec3& v){
    os << v.x() << " " << v.y() << " " << v.z() << std::endl;
    return os;
  }

 private:
  T x_{}, y_{}, z_{};

  T& xImpl() { return x_; }
  const T& yImpl() const { return y_; }
  T& yImpl() { return y_; }
};

class Transform {
 public:
  ARIA_PROP_BEGIN(public, public, , Vec3<float>, forward);
    ARIA_SUB_PROP(, float, x);
    ARIA_SUB_PROP(, float, y);
    ARIA_SUB_PROP(, float, z);
    ARIA_PROP_FUNC(public, , ., Func0);
    ARIA_PROP_FUNC(public, , ., Func1);
    ARIA_PROP_FUNC(public, , ., Func2);
    ARIA_PROP_FUNC(public, , ., Func3);
  ARIA_PROP_END;

 private:
  Vec3<double> forward_;

  [[nodiscard]] auto ARIA_PROP_GETTER(forward)() const {
    return Vec3<float>{float(forward_.x()),
                       float(forward_.y()),
                       float(forward_.z())};
  }
  void ARIA_PROP_SETTER(forward)(const auto& value) {
    forward_.x() = value.x();
    forward_.y() = value.y();
    forward_.z() = value.z();
  }
};

class TransformByRef {
public:
  ARIA_PROP_BEGIN(public, public, , Vec3<float>&, forward);
    ARIA_SUB_PROP(, float&, x);
    ARIA_SUB_PROP(, float&, y);
    ARIA_SUB_PROP(, float&, z);
    ARIA_PROP_FUNC(public, , ., Func0);
    ARIA_PROP_FUNC(public, , ., Func1);
    ARIA_PROP_FUNC(public, , ., Func2);
    ARIA_PROP_FUNC(public, , ., Func3);
  ARIA_PROP_END;

private:
  Vec3<float> forward_;

  [[nodiscard]] const Vec3<float>& ARIA_PROP_GETTER(forward)() const {
    return forward_;
  }
  [[nodiscard]] Vec3<float>& ARIA_PROP_GETTER(forward)() {
    return forward_;
  }
  void ARIA_PROP_SETTER(forward)(const auto& value) {
    forward_.x() = value.x();
    forward_.y() = value.y();
    forward_.z() = value.z();
  }
};

class Test1Arg {
public:
  ARIA_PROP(public, public, , int, test);

private:
  int base_ = 10;

  [[nodiscard]] int ARIA_PROP_GETTER(test)(int v) const {
    return base_ + v;
  }
  [[nodiscard]] int ARIA_PROP_GETTER(test)(int v) {
    return base_ + v;
  }
};

class Test2Args {
public:
  ARIA_PROP(public, public, , int, test);

private:
  int base_ = 10;

  [[nodiscard]] int ARIA_PROP_GETTER(test)(int v0, float v1) const {
    return base_ + v0 + static_cast<int>(std::floor(v1));
  }
  [[nodiscard]] int ARIA_PROP_GETTER(test)(int v0, float v1) {
    return base_ + v0 + static_cast<int>(std::floor(v1));
  }
};

class LBMD2Q9 {
public:
  ARIA_PROP(public, public, , int, f);

private:
  std::array<int, 9> f_;

  [[nodiscard]] int ARIA_PROP_GETTER(f)(const std::pair<int, int>& coord, auto q) {
    constexpr int testConstexpr = q;
    return f_[q];
  }
  void ARIA_PROP_SETTER(f)(const std::pair<int, int>& coord, auto q, int value) {
    constexpr int testConstexpr = q;
    f_[q] = value;
  }
};

} // namespace



TEST(Property, Operators) {
  // Get by copy
  {
    A a;
    const auto& y = a.x();

    EXPECT_EQ(a.x(), 0); // constructor
    a.x() = 1;
    EXPECT_EQ(a.x(), 1); // p = v
    a.x() = a.x() + 2;
    EXPECT_EQ(a.x(), 3); // p o v
    a.x() = 3 - a.x();
    EXPECT_EQ(a.x(), 0); // v o p
    a.x() += 4;
    EXPECT_EQ(a.x(), 4); // p o= v
    {
      int tmp = ++a.x();
      // int tmp = a.x()++;
      EXPECT_EQ(tmp, 5); // ++p return value
    }
    EXPECT_EQ(a.x(), 5); // ++p
    {
      int tmp = (++a.x()) += 1;
      EXPECT_EQ(tmp, 7); // o= return value
    }
    EXPECT_EQ(a.x(), 7); // ++p & o=
    {
      int tmp = (--a.x()) -= 1;
      EXPECT_EQ(tmp, 5); // o= return value
    }
    EXPECT_EQ(a.x(), 5); // --p & o=
    a.x() = -a.x();
    EXPECT_EQ(a.x(), -5); // -p positive to negative
    a.x() = -a.x();
    EXPECT_EQ(a.x(), 5); // -p negative to positive
    a.x() = a.x() + a.x();
    EXPECT_EQ(a.x(), 10); // p o p
    a.x() += a.x();
    EXPECT_EQ(a.x(), 20); // p o= p
    a.x() = (a.x() -= 10);
    EXPECT_EQ(a.x(), 10); // p = p
    a.x() = (a.x() /= 2);
    EXPECT_EQ(a.x(), 5); // p = p

    a.x() = 5;
    EXPECT_TRUE(a.x() == 5); // == & !=
    EXPECT_TRUE(5 == a.x());
    EXPECT_TRUE(a.x() == a.x());
    EXPECT_TRUE(a.x() == y);
    EXPECT_TRUE(y == a.x());

    EXPECT_TRUE(a.x()); // Implicit cast to bool
    EXPECT_TRUE(!A().x());
    EXPECT_TRUE(true || A().x());
    EXPECT_FALSE(a.x() && false);
    EXPECT_TRUE(a.x() || A().x());
    EXPECT_FALSE(a.x() && A().x());

    a.x() = 1; // Left and right shift
    EXPECT_EQ(a.x() << 3, 8);
    a.x() <<= 3;
    EXPECT_EQ(a.x(), 8);
    EXPECT_EQ(a.x() >> 2, 2);
    a.x() >>= 2;
    EXPECT_EQ(a.x(), 2);

    a.x() = 3; // Implicit cast to wider types
    EXPECT_EQ(a.x() / 2, 1);
    EXPECT_DOUBLE_EQ(a.x() / 1.5, 2);
    EXPECT_DOUBLE_EQ(a.x() / 2.0, 1.5);
    EXPECT_TRUE(a.x() != 3.1);
    EXPECT_TRUE(a.x() == 3.0);
    EXPECT_FALSE(a.x() == 3.1);
    EXPECT_FALSE(a.x() != 3.0);

    a.x() = 1; // Tricky implicit to narrower types
    EXPECT_TRUE(a.x() == uint16_t(1));
    EXPECT_TRUE(a.x() == uint8_t(1));
  }

  // Case between properties
  {
    A a;
    B b;

    a.x() = 2;
    b.x() = 3;

    EXPECT_EQ(a.x() + b.x(), 5);
    EXPECT_EQ(a.x() - b.x(), -1);
    EXPECT_EQ(a.x() += b.x(), 5);
    EXPECT_EQ(a.x() -= b.x(), 2);

    EXPECT_EQ(b.x() + a.x(), 5);
    EXPECT_EQ(b.x() - a.x(), 1);
    EXPECT_EQ(b.x() += a.x(), 5);
    EXPECT_EQ(b.x() -= a.x(), 3);

    EXPECT_TRUE(a.x() && b.x());
    EXPECT_FALSE(!a.x() && b.x());
    EXPECT_FALSE(a.x() && !b.x());

    EXPECT_TRUE(b.x() && a.x());
    EXPECT_FALSE(!b.x() && a.x());
    EXPECT_FALSE(b.x() && !a.x());
  }

  // std::ostream
  {
    A a;
    std::stringstream ss;
    ss << "Testing std::ostream: " << a << std::endl;
  }
}

TEST(Property, ForwardValueByReference) {
  {
    ChaosA a;



    { // Reference
      static_assert(property::detail::PropertyType<decltype(a.h())>);
      static_assert(property::detail::PropertyType<decltype(a.h().x())>);
      static_assert(property::detail::PropertyType<decltype(a.h().x().y())>);
      static_assert(property::detail::PropertyType<decltype(a.h().x().y().z())>);

      static_assert(!std::is_reference_v<decltype(a.h())>);
      static_assert(!std::is_reference_v<decltype(a.h().x())>);
      static_assert(!std::is_reference_v<decltype(a.h().x().y())>);
      static_assert(!std::is_reference_v<decltype(a.h().x().y().z())>);

      static_assert(!std::is_reference_v<decltype(a.h().value())>);
      static_assert(!std::is_reference_v<decltype(a.h().x().value())>);
      static_assert(!std::is_reference_v<decltype(a.h().x().y().value())>);
      static_assert(!std::is_reference_v<decltype(a.h().x().y().z().value())>);
    }



    auto expectA = [&] (const auto& v) {
      { // Explicit type
        HierarchyA::X tmpX = a.h().x();
        HierarchyA::X::Y tmpXY = tmpX.y();
        HierarchyA::X::Y tmpY = a.h().x().y();
        int tmpXYZ = tmpX.y().z();
        int tmpYZ = tmpY.z();

        EXPECT_TRUE(a.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.y_.z_ == v);
        EXPECT_TRUE(tmpXY.z_ == v);
        EXPECT_TRUE(tmpY.z_ == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        EXPECT_TRUE(a.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(!property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(!property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(!property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(!property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(!property::detail::PropertyType<decltype(tmpXYZ)>);
        static_assert(!property::detail::PropertyType<decltype(tmpYZ)>);

        static_assert(std::is_reference_v<decltype(tmpX.y())>);
        static_assert(std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);
      }

      { // auto
        auto tmpX = a.h().x();
        auto tmpXY = tmpX.y();
        auto tmpY = a.h().x().y();
        auto tmpXYZ = tmpX.y().z();
        auto tmpYZ = tmpY.z();

        EXPECT_TRUE(a.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(a.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(property::detail::PropertyType<decltype(tmpXYZ)>);
        static_assert(property::detail::PropertyType<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // const auto
        const auto tmpX = a.h().x();
        const auto tmpXY = tmpX.y();
        const auto tmpY = a.h().x().y();
        const auto tmpXYZ = tmpX.y().z();
        const auto tmpYZ = tmpY.z();

        EXPECT_TRUE(a.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(a.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y().z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXY.z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpY.z())>>);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXYZ)>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpYZ)>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // auto&&
        auto&& tmpX = a.h().x();
        auto&& tmpXY = tmpX.y();
        auto&& tmpY = a.h().x().y();
        auto&& tmpXYZ = tmpX.y().z();
        auto&& tmpYZ = tmpY.z();

        EXPECT_TRUE(a.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(a.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(property::detail::PropertyType<std::remove_reference_t<decltype(tmpXYZ)>>);
        static_assert(property::detail::PropertyType<std::remove_reference_t<decltype(tmpYZ)>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // const auto&
        const auto& tmpX = a.h().x();
        const auto& tmpXY = tmpX.y();
        const auto& tmpY = a.h().x().y();
        const auto& tmpXYZ = tmpX.y().z();
        const auto& tmpYZ = tmpY.z();

        EXPECT_TRUE(a.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(a.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y().z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXY.z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpY.z())>>);

        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpXYZ)>>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpYZ)>>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // const auto&&
        const auto&& tmpX = a.h().x();
        const auto&& tmpXY = tmpX.y();
        const auto&& tmpY = a.h().x().y();
        const auto&& tmpXYZ = tmpX.y().z();
        const auto&& tmpYZ = tmpY.z();

        EXPECT_TRUE(a.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(a.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y().z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXY.z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpY.z())>>);

        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpXYZ)>>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpYZ)>>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // decltype(auto)
        decltype(auto) tmpX = a.h().x();
        decltype(auto) tmpXY = tmpX.y();
        decltype(auto) tmpY = a.h().x().y();
        decltype(auto) tmpXYZ = tmpX.y().z();
        decltype(auto) tmpYZ = tmpY.z();

        EXPECT_TRUE(a.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(a.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(property::detail::PropertyType<decltype(tmpXYZ)>);
        static_assert(property::detail::PropertyType<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }
    };



    { // Construct and assignment
      a.h() = HierarchyA{};
      expectA(5);
      a.h() = HierarchyA{.x_{.y_{.z_ = 1}}};
      expectA(1);
      a.h().x() = HierarchyA::X{.y_{.z_ = 2}};
      expectA(2);
      a.h().x().y() = HierarchyA::X::Y{.z_ = 3};
      expectA(3);
      a.h().x().y().z() = 5;
      expectA(5);
    }

    { // Set
      a.h().x().y().z() = 0;
      expectA(0);

      { // Copy
        auto tmp = a.h();
        tmp.x().y().z() = 1;
        expectA(1);
      }
      {
        auto tmp = a.h().x();
        tmp.y().z() = 2;
        expectA(2);
      }
      {
        auto tmp = a.h().x().y();
        tmp.z() = 3;
        expectA(3);
      }
      {
        auto tmp = a.h().x().y().z();
        tmp = 4;
        expectA(4);
      }

      { // Reference
        auto&& tmp = a.h();
        tmp.x().y().z() = 1;
        expectA(1);
      }
      {
        auto&& tmp = a.h().x();
        tmp.y().z() = 2;
        expectA(2);
      }
      {
        auto&& tmp = a.h().x().y();
        tmp.z() = 3;
        expectA(3);
      }
      {
        auto&& tmp = a.h().x().y().z();
        tmp = 4;
        expectA(4);
      }
    }



    { // Tree-like hierarchies
      EXPECT_TRUE(a.h().x1().y0() == 10);
      EXPECT_TRUE(a.h().x1().y1() == 20);
    }
  }
}

TEST(Property, ForwardValueByCopy) {
  {
    ChaosB b;



    { // Reference
      static_assert(property::detail::PropertyType<decltype(b.h())>);
      static_assert(property::detail::PropertyType<decltype(b.h().x())>);
      static_assert(property::detail::PropertyType<decltype(b.h().x().y())>);
      static_assert(property::detail::PropertyType<decltype(b.h().x().y().z())>);

      static_assert(!std::is_reference_v<decltype(b.h())>);
      static_assert(!std::is_reference_v<decltype(b.h().x())>);
      static_assert(!std::is_reference_v<decltype(b.h().x().y())>);
      static_assert(!std::is_reference_v<decltype(b.h().x().y().z())>);

      static_assert(!std::is_reference_v<decltype(b.h().value())>);
      static_assert(!std::is_reference_v<decltype(b.h().x().value())>);
      static_assert(!std::is_reference_v<decltype(b.h().x().y().value())>);
      static_assert(!std::is_reference_v<decltype(b.h().x().y().z().value())>);
    }



    auto expectB = [&] (const auto& v) {
      { // Explicit type
        HierarchyB::X tmpX = b.h().x();
        HierarchyB::X::Y tmpXY = tmpX.y();
        HierarchyB::X::Y tmpY = b.h().x().y();
        int tmpXYZ = tmpX.y().z();
        int tmpYZ = tmpY.z();

        EXPECT_TRUE(b.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.y_.z_ == v);
        EXPECT_TRUE(tmpXY.z_ == v);
        EXPECT_TRUE(tmpY.z_ == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        EXPECT_TRUE(b.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(!property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(!property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(!property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(!property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(!property::detail::PropertyType<decltype(tmpXYZ)>);
        static_assert(!property::detail::PropertyType<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);
      }

      { // auto
        auto tmpX = b.h().x();
        auto tmpXY = tmpX.y();
        auto tmpY = b.h().x().y();
        auto tmpXYZ = tmpX.y().z();
        auto tmpYZ = tmpY.z();

        EXPECT_TRUE(b.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(b.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(property::detail::PropertyType<decltype(tmpXYZ)>);
        static_assert(property::detail::PropertyType<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // const auto
        const auto tmpX = b.h().x();
        const auto tmpXY = tmpX.y();
        const auto tmpY = b.h().x().y();
        const auto tmpXYZ = tmpX.y().z();
        const auto tmpYZ = tmpY.z();

        EXPECT_TRUE(b.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(b.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y().z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXY.z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpY.z())>>);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXYZ)>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpYZ)>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // auto&&
        auto&& tmpX = b.h().x();
        auto&& tmpXY = tmpX.y();
        auto&& tmpY = b.h().x().y();
        auto&& tmpXYZ = tmpX.y().z();
        auto&& tmpYZ = tmpY.z();

        EXPECT_TRUE(b.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(b.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(property::detail::PropertyType<std::remove_reference_t<decltype(tmpXYZ)>>);
        static_assert(property::detail::PropertyType<std::remove_reference_t<decltype(tmpYZ)>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // const auto&
        const auto& tmpX = b.h().x();
        const auto& tmpXY = tmpX.y();
        const auto& tmpY = b.h().x().y();
        const auto& tmpXYZ = tmpX.y().z();
        const auto& tmpYZ = tmpY.z();

        EXPECT_TRUE(b.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(b.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y().z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXY.z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpY.z())>>);

        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpXYZ)>>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpYZ)>>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // const auto&&
        const auto&& tmpX = b.h().x();
        const auto&& tmpXY = tmpX.y();
        const auto&& tmpY = b.h().x().y();
        const auto&& tmpXYZ = tmpX.y().z();
        const auto&& tmpYZ = tmpY.z();

        EXPECT_TRUE(b.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(b.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y().z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXY.z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpY.z())>>);

        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpXYZ)>>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpYZ)>>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // decltype(auto)
        decltype(auto) tmpX = b.h().x();
        decltype(auto) tmpXY = tmpX.y();
        decltype(auto) tmpY = b.h().x().y();
        decltype(auto) tmpXYZ = tmpX.y().z();
        decltype(auto) tmpYZ = tmpY.z();

        EXPECT_TRUE(b.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(b.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(property::detail::PropertyType<decltype(tmpXYZ)>);
        static_assert(property::detail::PropertyType<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }
    };



    { // Construct and assignment
      b.h() = HierarchyB{};
      expectB(5);
      b.h() = HierarchyB{.x_{.y_{.z_ = 1}}};
      expectB(1);
      // b.h().x() = HierarchyB::X{.y_{.z_ = 2}};
      // expectB(2);
      // b.h().x().y() = HierarchyB::X::Y{.z_ = 3};
      // expectB(3);
      // b.h().x().y().z() = 5;
      // expectB(5);
    }

    { // Set
      // b.h().x().y().z() = 0;
      // expectB(0);

      { // Copy
        // auto tmp = b.h();
        // tmp.x().y().z() = 1;
        // expectB(1);
      }
      {
        // auto tmp = b.h().x();
        // tmp.y().z() = 2;
        // expectB(2);
      }
      {
        // auto tmp = b.h().x().y();
        // tmp.z() = 3;
        // expectB(3);
      }
      {
        // auto tmp = b.h().x().y().z();
        // tmp = 4;
        // expectB(4);
      }

      { // Reference
        // auto&& tmp = b.h();
        // tmp.x().y().z() = 1;
        // expectB(1);
      }
      {
        // auto&& tmp = b.h().x();
        // tmp.y().z() = 2;
        // expectB(2);
      }
      {
        // auto&& tmp = b.h().x().y();
        // tmp.z() = 3;
        // expectB(3);
      }
      {
        // auto&& tmp = b.h().x().y().z();
        // tmp = 4;
        // expectB(4);
      }
    }



    { // Tree-like hierarchies
      EXPECT_TRUE(b.h().x1().y0() == 10);
      EXPECT_TRUE(b.h().x1().y1() == 20);
    }
  }
}

TEST(Property, ForwardValueByHybridReferenceAndCopy) {
  {
    ChaosC c;



    { // Reference
      static_assert(property::detail::PropertyType<decltype(c.h())>);
      static_assert(property::detail::PropertyType<decltype(c.h().x())>);
      static_assert(property::detail::PropertyType<decltype(c.h().x().y())>);
      static_assert(property::detail::PropertyType<decltype(c.h().x().y().z())>);

      static_assert(!std::is_reference_v<decltype(c.h())>);
      static_assert(!std::is_reference_v<decltype(c.h().x())>);
      static_assert(!std::is_reference_v<decltype(c.h().x().y())>);
      static_assert(!std::is_reference_v<decltype(c.h().x().y().z())>);

      static_assert(!std::is_reference_v<decltype(c.h().value())>);
      static_assert(!std::is_reference_v<decltype(c.h().x().value())>);
      static_assert(!std::is_reference_v<decltype(c.h().x().y().value())>);
      static_assert(!std::is_reference_v<decltype(c.h().x().y().z().value())>);
    }



    auto expectC = [&] (const auto& v) {
      { // Explicit type
        HierarchyC::X tmpX = c.h().x();
        HierarchyC::X::Y tmpXY = tmpX.y();
        HierarchyC::X::Y tmpY = c.h().x().y();
        int tmpXYZ = tmpX.y().z();
        int tmpYZ = tmpY.z();

        EXPECT_TRUE(c.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.y_.z_ == v);
        EXPECT_TRUE(tmpXY.z_ == v);
        EXPECT_TRUE(tmpY.z_ == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        EXPECT_TRUE(c.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(!property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(!property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(!property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(!property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(!property::detail::PropertyType<decltype(tmpXYZ)>);
        static_assert(!property::detail::PropertyType<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);
      }

      { // auto
        auto tmpX = c.h().x();
        auto tmpXY = tmpX.y();
        auto tmpY = c.h().x().y();
        auto tmpXYZ = tmpX.y().z();
        auto tmpYZ = tmpY.z();

        EXPECT_TRUE(c.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(c.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(property::detail::PropertyType<decltype(tmpXYZ)>);
        static_assert(property::detail::PropertyType<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // const auto
        const auto tmpX = c.h().x();
        const auto tmpXY = tmpX.y();
        const auto tmpY = c.h().x().y();
        const auto tmpXYZ = tmpX.y().z();
        const auto tmpYZ = tmpY.z();

        EXPECT_TRUE(c.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(c.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y().z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXY.z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpY.z())>>);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXYZ)>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpYZ)>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // auto&&
        auto&& tmpX = c.h().x();
        auto&& tmpXY = tmpX.y();
        auto&& tmpY = c.h().x().y();
        auto&& tmpXYZ = tmpX.y().z();
        auto&& tmpYZ = tmpY.z();

        EXPECT_TRUE(c.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(c.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(property::detail::PropertyType<std::remove_reference_t<decltype(tmpXYZ)>>);
        static_assert(property::detail::PropertyType<std::remove_reference_t<decltype(tmpYZ)>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // const auto&
        const auto& tmpX = c.h().x();
        const auto& tmpXY = tmpX.y();
        const auto& tmpY = c.h().x().y();
        const auto& tmpXYZ = tmpX.y().z();
        const auto& tmpYZ = tmpY.z();

        EXPECT_TRUE(c.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(c.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y().z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXY.z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpY.z())>>);

        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpXYZ)>>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpYZ)>>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // const auto&&
        const auto&& tmpX = c.h().x();
        const auto&& tmpXY = tmpX.y();
        const auto&& tmpY = c.h().x().y();
        const auto&& tmpXYZ = tmpX.y().z();
        const auto&& tmpYZ = tmpY.z();

        EXPECT_TRUE(c.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(c.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpX.y().z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpXY.z())>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<decltype(tmpY.z())>>);

        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpXYZ)>>>);
        static_assert(property::detail::PropertyType<std::remove_const_t<std::remove_reference_t<decltype(tmpYZ)>>>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }

      { // decltype(auto)
        decltype(auto) tmpX = c.h().x();
        decltype(auto) tmpXY = tmpX.y();
        decltype(auto) tmpY = c.h().x().y();
        decltype(auto) tmpXYZ = tmpX.y().z();
        decltype(auto) tmpYZ = tmpY.z();

        EXPECT_TRUE(c.h_.x_.y_.z_ == v);
        EXPECT_TRUE(tmpX.value().y_.z_ == v);
        EXPECT_TRUE(tmpXY.value().z_ == v);
        EXPECT_TRUE(tmpY.value().z_ == v);
        EXPECT_TRUE(tmpXYZ.value() == v);
        EXPECT_TRUE(tmpYZ.value() == v);

        EXPECT_TRUE(c.h().x().y().z() == v);
        EXPECT_TRUE(tmpX.y().z() == v);
        EXPECT_TRUE(tmpXY.z() == v);
        EXPECT_TRUE(tmpY.z() == v);
        EXPECT_TRUE(tmpXYZ == v);
        EXPECT_TRUE(tmpYZ == v);

        static_assert(property::detail::PropertyType<decltype(tmpX.y())>);
        static_assert(property::detail::PropertyType<decltype(tmpX.y().z())>);
        static_assert(property::detail::PropertyType<decltype(tmpXY.z())>);
        static_assert(property::detail::PropertyType<decltype(tmpY.z())>);

        static_assert(property::detail::PropertyType<decltype(tmpXYZ)>);
        static_assert(property::detail::PropertyType<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ)>);
        static_assert(!std::is_reference_v<decltype(tmpYZ)>);

        static_assert(!std::is_reference_v<decltype(tmpX.y().value())>);
        static_assert(!std::is_reference_v<decltype(tmpX.y().z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpXY.z().value())>);
        static_assert(!std::is_reference_v<decltype(tmpY.z().value())>);

        static_assert(!std::is_reference_v<decltype(tmpXYZ.value())>);
        static_assert(!std::is_reference_v<decltype(tmpYZ.value())>);
      }
    };



    { // Construct and assignment
      c.h() = HierarchyC{};
      expectC(5);
      c.h() = HierarchyC{.x_{.y_{.z_ = 1}}};
      expectC(1);
      c.h().x() = HierarchyC::X{.y_{.z_ = 2}};
      expectC(2);
      // c.h().x().y() = HierarchyC::X::Y{.z_ = 3};
      // expectC(3);
      // c.h().x().y().z() = 5;
      // expectC(5);
    }

    { // Set
      // c.h().x().y().z() = 0;
      // expectC(0);

      { // Copy
        // auto tmp = c.h();
        // tmp.x().y().z() = 1;
        // expectC(1);
      }
      {
        // auto tmp = c.h().x();
        // tmp.y().z() = 2;
        // expectC(2);
      }
      {
        // auto tmp = c.h().x().y();
        // tmp.z() = 3;
        // expectC(3);
      }
      {
        // auto tmp = c.h().x().y().z();
        // tmp = 4;
        // expectC(4);
      }

      { // Reference
        // auto&& tmp = c.h();
        // tmp.x().y().z() = 1;
        // expectC(1);
      }
      {
        // auto&& tmp = c.h().x();
        // tmp.y().z() = 2;
        // expectC(2);
      }
      {
        // auto&& tmp = c.h().x().y();
        // tmp.z() = 3;
        // expectC(3);
      }
      {
        // auto&& tmp = c.h().x().y().z();
        // tmp = 4;
        // expectC(4);
      }
    }



    { // Tree-like hierarchies
      EXPECT_TRUE(c.h().x1().y0() == 10);
      EXPECT_TRUE(c.h().x1().y1() == 20);
    }
  }
}

TEST(Property, MemberProperties) {
  { // Easy
    ChaosListA la;

    auto expectA = [&] (const auto& v) {
      EXPECT_TRUE(la.a().b().c().d().e().f().g() == v);
      EXPECT_TRUE(la.a().value().b().c().d().e().f().g() == v);
      EXPECT_TRUE(la.a().value().b().c().d().value().e().f().g() == v);
      EXPECT_TRUE(la.a().value().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(la.a().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(la.a().b().c().d().e().f().g().value() == v);
    };

    expectA(5);

    // la.a().b().c().d().e().f().g() = 0;
    // expectA(0);

    // la.a().b().c().d().e().f() = ChaosListA::A::B::C::D::E::F{.g_ = 0};
    // expectA(0);

    // la.a().b().c().d().e() = ChaosListA::A::B::C::D::E{.f_{.g_ = 0}};
    // expectA(0);

    // la.a().b().c().d() = ChaosListA::A::B::C::D{.e_{.f_{.g_ = 0}}};
    // expectA(0);

    // la.a().b().c() = ChaosListA::A::B::C{.d_{.e_{.f_{.g_ = 0}}}};
    // expectA(0);

    la.a().b() = ChaosListA::A::B{.c_{.d_{.e_{.f_{.g_ = 0}}}}};
    expectA(0);

    la.a() = ChaosListA::A{.b_{.c_{.d_{.e_{.f_{.g_ = -1}}}}}};
    expectA(-1);

    la = ChaosListA{.a_{.b_{.c_{.d_{.e_{.f_{.g_ = -2}}}}}}};
    expectA(-2);
  }

  { // Normal
    ChaosListB lb;

    auto expectB = [&] (const auto& v) {
      EXPECT_TRUE(lb.a().a().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lb.a().a().value().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lb.a().a().value().b().c().d().value().e().f().g() == v);
      EXPECT_TRUE(lb.a().a().value().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lb.a().a().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lb.a().a().b().c().d().e().f().g().value() == v);

      EXPECT_TRUE(lb.a().value().a().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lb.a().value().a().value().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lb.a().value().a().value().b().c().d().value().e().f().g() == v);
      EXPECT_TRUE(lb.a().value().a().value().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lb.a().value().a().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lb.a().value().a().b().c().d().e().f().g().value() == v);
    };

    expectB(5);

    // lb.a().a().b().c().d().e().f().g() = 0;
    // expectB(0);

    // lb.a().a().b().c().d().e().f() = ChaosListA::A::B::C::D::E::F{.g_ = 0};
    // expectB(0);

    // lb.a().a().b().c().d().e() = ChaosListA::A::B::C::D::E{.f_{.g_ = 0}};
    // expectB(0);

    // lb.a().a().b().c().d() = ChaosListA::A::B::C::D{.e_{.f_{.g_ = 0}}};
    // expectB(0);

    // lb.a().a().b().c() = ChaosListA::A::B::C{.d_{.e_{.f_{.g_ = 0}}}};
    // expectB(0);

    lb.a().a().b() = ChaosListA::A::B{.c_{.d_{.e_{.f_{.g_ = 0}}}}};
    expectB(0);

    lb.a().a() = ChaosListA::A{.b_{.c_{.d_{.e_{.f_{.g_ = -1}}}}}};
    expectB(-1);

    lb.a() = ChaosListA{.a_{.b_{.c_{.d_{.e_{.f_{.g_ = -2}}}}}}};
    expectB(-2);

    lb = ChaosListB{.a_{.a_{.b_{.c_{.d_{.e_{.f_{.g_ = -3}}}}}}}};
    expectB(-3);
  }

  { // Hard
    ChaosListC lc;

    auto expectC = [&] (const auto& v) {
      EXPECT_TRUE(lc.b().a().a().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lc.b().a().a().value().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lc.b().a().a().value().b().c().d().value().e().f().g() == v);
      EXPECT_TRUE(lc.b().a().a().value().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lc.b().a().a().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lc.b().a().a().b().c().d().e().f().g().value() == v);

      EXPECT_TRUE(lc.b().a().value().a().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lc.b().a().value().a().value().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lc.b().a().value().a().value().b().c().d().value().e().f().g() == v);
      EXPECT_TRUE(lc.b().a().value().a().value().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lc.b().a().value().a().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lc.b().a().value().a().b().c().d().e().f().g().value() == v);

      EXPECT_TRUE(lc.b().value().a().a().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lc.b().value().a().a().value().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lc.b().value().a().a().value().b().c().d().value().e().f().g() == v);
      EXPECT_TRUE(lc.b().value().a().a().value().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lc.b().value().a().a().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lc.b().value().a().a().b().c().d().e().f().g().value() == v);

      EXPECT_TRUE(lc.b().value().a().value().a().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lc.b().value().a().value().a().value().b().c().d().e().f().g() == v);
      EXPECT_TRUE(lc.b().value().a().value().a().value().b().c().d().value().e().f().g() == v);
      EXPECT_TRUE(lc.b().value().a().value().a().value().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lc.b().value().a().value().a().b().c().d().value().e().f().g().value() == v);
      EXPECT_TRUE(lc.b().value().a().value().a().b().c().d().e().f().g().value() == v);
    };

    expectC(5);

    // lc.b().a().a().b().c().d().e().f().g() = 0;
    // expectC(0);

    // lc.b().a().a().b().c().d().e().f() = ChaosListA::A::B::C::D::E::F{.g_ = 0};
    // expectC(0);

    // lc.b().a().a().b().c().d().e() = ChaosListA::A::B::C::D::E{.f_{.g_ = 0}};
    // expectC(0);

    // lc.b().a().a().b().c().d() = ChaosListA::A::B::C::D{.e_{.f_{.g_ = 0}}};
    // expectC(0);

    // lc.b().a().a().b().c() = ChaosListA::A::B::C{.d_{.e_{.f_{.g_ = 0}}}};
    // expectC(0);

    lc.b().a().a().b() = ChaosListA::A::B{.c_{.d_{.e_{.f_{.g_ = 0}}}}};
    expectC(0);

    lc.b().a().a() = ChaosListA::A{.b_{.c_{.d_{.e_{.f_{.g_ = -1}}}}}};
    expectC(-1);

    lc.b().a() = ChaosListA{.a_{.b_{.c_{.d_{.e_{.f_{.g_ = -2}}}}}}};
    expectC(-2);

    lc.b() = ChaosListB{.a_{.a_{.b_{.c_{.d_{.e_{.f_{.g_ = -3}}}}}}}};
    expectC(-3);

    lc = ChaosListC{.b_{.a_{.a_{.b_{.c_{.d_{.e_{.f_{.g_ = -4}}}}}}}}};
    expectC(-4);
  }
}

TEST(Property, BooleanVector) {
  VectorBool v;
  static_assert(std::is_same_v<decltype(v.v().value()), bool>);

  // v.vs() = std::vector<bool>{};

  EXPECT_TRUE(v.v() == false);
  EXPECT_TRUE(v.vs()[0] == false);

  v.v() = true;
  EXPECT_TRUE(v.v() == true);
  EXPECT_TRUE(v.vs()[0] == true);

  v.v() = false;
  EXPECT_TRUE(v.v() == false);
  EXPECT_TRUE(v.vs()[0] == false);
}

TEST(Property, Callable) {
  Func func;

  int res{func.f0()(5)};
  EXPECT_TRUE(res == 6);

  func.f1()(6);
}

TEST(Property, Vec3) {
  using Vec3f = Vec3<float>;

  Vec3f v;
  EXPECT_FLOAT_EQ(v.x(), 0);
  EXPECT_FLOAT_EQ(v.y(), 0);
  EXPECT_FLOAT_EQ(v.z(), 0);

  v.x() = 1.1f;
  v.y() = 2.2f;
  v.z() = 3.3f;
  EXPECT_FLOAT_EQ(v.x(), 1.1f);
  EXPECT_FLOAT_EQ(v.y(), 2.2f);
  EXPECT_FLOAT_EQ(v.z(), 3.3f);

  v.xx() = 1.1f;
  v.yy() = 2.2f;
  v.zz() = 3.3f;
  EXPECT_FLOAT_EQ(v.xx(), 1.1f);
  EXPECT_FLOAT_EQ(v.yy(), 2.2f);
  EXPECT_FLOAT_EQ(v.zz(), 3.3f);
}

TEST(Property, Auto) {
  {
    Vec3<float> v;
    auto x = Auto(v.x());
    decltype(auto) y = Auto(v.x());
    static_assert(std::is_same_v<decltype(x), float>);
    static_assert(std::is_same_v<decltype(y), float&>);
  }

  {
    ChaosListC lc;
    auto x = Auto(lc.b().a().a().b().c().d().e().f().g());
    decltype(auto) y = Auto(lc.b().a().a().b().c().d().e().f().g());
    static_assert(std::is_same_v<decltype(x), int>);
    static_assert(std::is_same_v<decltype(y), int>);
  }
}

TEST(Property, Function) {
  Transform t0;
  const Transform t1;

  {
    auto v = t0.forward().Func0(0.0f);
    static_assert(std::is_same_v<decltype(v), float>);
  }

  {
    t0.forward().Func1(1.0f);
  }

  {
    auto v = t0.forward().Func2(-0.0f);
    static_assert(std::is_same_v<decltype(v), float>);
  }

  {
    t0.forward().Func3(-1.0f);
  }

  // {
  //   auto v = t1.forward().Func0(0.0f);
  //   static_assert(std::is_same_v<decltype(v), float>);
  // }

  // {
  //   t1.forward().Func1(1.0f);
  // }

  {
    auto v = t1.forward().Func2(-0.0f);
    static_assert(std::is_same_v<decltype(v), float>);
  }

  {
    t1.forward().Func3(-1.0f);
  }

  // TODO: How to use concepts to force the non-const version is never called?

  // Auto.
  {
    static_assert(std::is_same_v<decltype(Auto(t0.forward())), Vec3<float>>);
    auto f0 = Auto(t0.forward());
    static_assert(std::is_same_v<decltype(f0), Vec3<float>>);

    static_assert(std::is_same_v<decltype(Auto(t1.forward())), Vec3<float>>);
    auto f1 = Auto(t1.forward());
    static_assert(std::is_same_v<decltype(f1), Vec3<float>>);
  }

  // Smart reference.
  {
    Property auto f0 = t0.forward();

    f0 = {1.11f, 2.22f, 3.33f};
    EXPECT_FLOAT_EQ(t0.forward().x(), 1.11f);
    EXPECT_FLOAT_EQ(t0.forward().y(), 2.22f);
    EXPECT_FLOAT_EQ(t0.forward().z(), 3.33f);
    EXPECT_FLOAT_EQ(f0.x(), 1.11f);
    EXPECT_FLOAT_EQ(f0.y(), 2.22f);
    EXPECT_FLOAT_EQ(f0.z(), 3.33f);

    f0.x() = 3.3f;
    EXPECT_FLOAT_EQ(t0.forward().x(), 3.3f);
    EXPECT_FLOAT_EQ(f0.x(), 3.3f);
    f0.y() = 2.2f;
    EXPECT_FLOAT_EQ(t0.forward().y(), 2.2f);
    EXPECT_FLOAT_EQ(f0.y(), 2.2f);
    f0.z() = 1.1f;
    EXPECT_FLOAT_EQ(t0.forward().z(), 1.1f);
    EXPECT_FLOAT_EQ(f0.z(), 1.1f);
  }
}

TEST(Property, ReferenceProperties) {
  TransformByRef t0;
  const TransformByRef t1;

  t0.forward() = Vec3<float>{1.1f, 2.2f, 3.3f};
  EXPECT_FLOAT_EQ(t0.forward().x(), 1.1f);
  EXPECT_FLOAT_EQ(t0.forward().y(), 2.2f);
  EXPECT_FLOAT_EQ(t0.forward().z(), 3.3f);

  t0.forward() = {1.11f, 2.22f, 3.33f};
  EXPECT_FLOAT_EQ(t0.forward().x(), 1.11f);
  EXPECT_FLOAT_EQ(t0.forward().y(), 2.22f);
  EXPECT_FLOAT_EQ(t0.forward().z(), 3.33f);

  t0.forward().x() = 3.3f;
  EXPECT_FLOAT_EQ(t0.forward().x(), 3.3f);
  t0.forward().y() = 2.2f;
  EXPECT_FLOAT_EQ(t0.forward().y(), 2.2f);
  t0.forward().z() = 1.1f;
  EXPECT_FLOAT_EQ(t0.forward().z(), 1.1f);

  // t1.forward() = Vec3<float>{1.1f, 2.2f, 3.3f};
  // t1.forward() = {1.11f, 2.22f, 3.33f};

  // t1.forward().x() = 3.3f;
  // t1.forward().y() = 2.2f;
  // t1.forward().z() = 1.1f;

  // Auto.
  {
    static_assert(std::is_same_v<decltype(Auto(t0.forward())), Vec3<float>>);
    auto f0 = Auto(t0.forward());
    static_assert(std::is_same_v<decltype(f0), Vec3<float>>);

    static_assert(std::is_same_v<decltype(Auto(t1.forward())), Vec3<float>>);
    auto f1 = Auto(t1.forward());
    static_assert(std::is_same_v<decltype(f1), Vec3<float>>);
  }

  // Smart reference.
  {
    Property auto f0 = t0.forward();

    f0 = {1.11f, 2.22f, 3.33f};
    EXPECT_FLOAT_EQ(t0.forward().x(), 1.11f);
    EXPECT_FLOAT_EQ(t0.forward().y(), 2.22f);
    EXPECT_FLOAT_EQ(t0.forward().z(), 3.33f);
    EXPECT_FLOAT_EQ(f0.x(), 1.11f);
    EXPECT_FLOAT_EQ(f0.y(), 2.22f);
    EXPECT_FLOAT_EQ(f0.z(), 3.33f);

    f0.x() = 3.3f;
    EXPECT_FLOAT_EQ(t0.forward().x(), 3.3f);
    EXPECT_FLOAT_EQ(f0.x(), 3.3f);
    f0.y() = 2.2f;
    EXPECT_FLOAT_EQ(t0.forward().y(), 2.2f);
    EXPECT_FLOAT_EQ(f0.y(), 2.2f);
    f0.z() = 1.1f;
    EXPECT_FLOAT_EQ(t0.forward().z(), 1.1f);
    EXPECT_FLOAT_EQ(f0.z(), 1.1f);
  }
}

TEST(Property, PropertyArguments) {
  {
    Test1Arg t0;
    const Test1Arg t1;

    auto a0 = t0.test();
    auto a1 = t1.test();
    auto b0 = a0.args(9);
    auto b1 = a1.args(9);
    auto c0 = b0.args();
    auto c1 = b1.args();
    auto d0 = c0.args(9);
    auto d1 = c1.args(9);
    auto e0 = d0.args("Fake");
    auto e1 = d1.args("Fake");

    static_assert(property::detail::PropertyType<decltype(a0)>);
    static_assert(property::detail::PropertyType<decltype(a1)>);
    static_assert(property::detail::PropertyType<decltype(b0)>);
    static_assert(property::detail::PropertyType<decltype(b1)>);
    static_assert(property::detail::PropertyType<decltype(c0)>);
    static_assert(property::detail::PropertyType<decltype(c1)>);
    static_assert(property::detail::PropertyType<decltype(d0)>);
    static_assert(property::detail::PropertyType<decltype(d1)>);
    static_assert(property::detail::PropertyType<decltype(e0)>);
    static_assert(property::detail::PropertyType<decltype(e1)>);
    static_assert(std::is_same_v<decltype(a0), decltype(c0)>);
    static_assert(std::is_same_v<decltype(a1), decltype(c1)>);
    static_assert(std::is_same_v<decltype(b0), decltype(d0)>);
    static_assert(std::is_same_v<decltype(b1), decltype(d1)>);

    EXPECT_EQ(t0.test(9), 19);
    EXPECT_EQ(t1.test(9), 19);
    EXPECT_EQ(t0.test().args(9), 19);
    EXPECT_EQ(t1.test().args(9), 19);
    EXPECT_EQ(b0, 19);
    EXPECT_EQ(b1, 19);
    EXPECT_EQ(d0, 19);
    EXPECT_EQ(d1, 19);
  }

  {
    Test2Args t0;
    const Test2Args t1;

    auto a0 = t0.test();
    auto a1 = t1.test();
    auto b0 = a0.args(9, 1.999F);
    auto b1 = a1.args(9, 1.999F);
    auto c0 = b0.args();
    auto c1 = b1.args();
    auto d0 = c0.args(9, 1.999F);
    auto d1 = c1.args(9, 1.999F);
    auto e0 = d0.args("Fake");
    auto e1 = d1.args("Fake");

    static_assert(property::detail::PropertyType<decltype(a0)>);
    static_assert(property::detail::PropertyType<decltype(a1)>);
    static_assert(property::detail::PropertyType<decltype(b0)>);
    static_assert(property::detail::PropertyType<decltype(b1)>);
    static_assert(property::detail::PropertyType<decltype(c0)>);
    static_assert(property::detail::PropertyType<decltype(c1)>);
    static_assert(property::detail::PropertyType<decltype(d0)>);
    static_assert(property::detail::PropertyType<decltype(d1)>);
    static_assert(property::detail::PropertyType<decltype(e0)>);
    static_assert(property::detail::PropertyType<decltype(e1)>);
    static_assert(std::is_same_v<decltype(a0), decltype(c0)>);
    static_assert(std::is_same_v<decltype(a1), decltype(c1)>);
    static_assert(std::is_same_v<decltype(b0), decltype(d0)>);
    static_assert(std::is_same_v<decltype(b1), decltype(d1)>);

    EXPECT_EQ(t0.test(9, 1.999F), 20);
    EXPECT_EQ(t1.test(9, 1.999F), 20);
    EXPECT_EQ(t0.test().args(9, 1.999F), 20);
    EXPECT_EQ(t1.test().args(9, 1.999F), 20);
    EXPECT_EQ(b0, 20);
    EXPECT_EQ(b1, 20);
    EXPECT_EQ(d0, 20);
    EXPECT_EQ(d1, 20);
  }

  {
    LBMD2Q9 lbm;

    lbm.f(std::make_pair(0, 0), 0_I) = 9;
    lbm.f(std::make_pair(0, 0), 1_I) = 8;
    lbm.f(std::make_pair(0, 0), 2_I) = 7;
    lbm.f(std::make_pair(0, 0), 3_I) = 6;
    lbm.f(std::make_pair(0, 0), 4_I) = 5;
    lbm.f(std::make_pair(0, 0), 5_I) = 4;
    lbm.f(std::make_pair(0, 0), 6_I) = 3;
    lbm.f(std::make_pair(0, 0), 7_I) = 2;
    lbm.f(std::make_pair(0, 0), 8_I) = 1;

    EXPECT_EQ(lbm.f(std::make_pair(0, 0), 0_I), 9);
    EXPECT_EQ(lbm.f(std::make_pair(0, 0), 1_I), 8);
    EXPECT_EQ(lbm.f(std::make_pair(0, 0), 2_I), 7);
    EXPECT_EQ(lbm.f(std::make_pair(0, 0), 3_I), 6);
    EXPECT_EQ(lbm.f(std::make_pair(0, 0), 4_I), 5);
    EXPECT_EQ(lbm.f(std::make_pair(0, 0), 5_I), 4);
    EXPECT_EQ(lbm.f(std::make_pair(0, 0), 6_I), 3);
    EXPECT_EQ(lbm.f(std::make_pair(0, 0), 7_I), 2);
    EXPECT_EQ(lbm.f(std::make_pair(0, 0), 8_I), 1);
  }
}

// clang-format on

} // namespace ARIA
