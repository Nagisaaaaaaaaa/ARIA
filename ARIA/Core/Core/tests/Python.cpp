#include "ARIA/Python.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

int add(int x, int y) {
  return x + y;
}

float add(float x, float y) {
  return x + y;
}

std::vector<int> add(const std::vector<int> &a, const std::vector<int> &b) {
  size_t size = a.size();
  ARIA_ASSERT(size == b.size());

  std::vector<int> c(size);
  for (size_t i = 0; i < size; ++i)
    c[i] = a[i] + b[i];

  return c;
}

std::vector<int> add0(const std::vector<int> &a, const std::vector<int> &b) {
  return add(a, b);
}

//
//
//
struct ARIATestPython_GrandParent {
  virtual ~ARIATestPython_GrandParent() = default;

  virtual int value() = 0;
};

struct ARIATestPython_Parent : public ARIATestPython_GrandParent {
  virtual ~ARIATestPython_Parent() = default;

  int value() override { return 1; }
};

struct ARIATestPython_Child final : public ARIATestPython_Parent {
  virtual ~ARIATestPython_Child() = default;

  int value() final { return 2; }
};

//
//
//
struct ARIATestPython_OverloadWithConst {
  int value() const { return 0; }

  int value() { return 1; }
};

struct ARIATestPython_OverloadWithParameters {
  int value(int) { return 0; }

  std::string value(double) { return "0"; }

  std::vector<std::string> value(int, double) { return {"0"}; }
};

struct ARIATestPython_ManyOverloads {
  using str = std::string;

  std::vector<bool> member;

  ARIATestPython_ManyOverloads() {}

  ARIATestPython_ManyOverloads(const int &v0) {}

  ARIATestPython_ManyOverloads(const int &v0, const str &v1) {}

  ARIATestPython_ManyOverloads(int v0, str v1, int v2) {}

  ARIATestPython_ManyOverloads(int v0, str v1, int v2, str v3) {}

  ARIATestPython_ManyOverloads(int v0, str v1, int v2, str v3, int v4) {}

  ARIATestPython_ManyOverloads(int v0, str v1, int v2, str v3, int v4, str v5) {}

  ARIATestPython_ManyOverloads(int v0, str v1, int v2, str v3, int v4, str v5, int v6) {}

  ARIATestPython_ManyOverloads(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7) {}

  ARIATestPython_ManyOverloads(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8) {}

  ARIATestPython_ManyOverloads(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8, str v9) {}

  std::vector<bool> F() const { return {}; }

  std::vector<bool> F(const int &v0) const { return {}; }

  std::vector<bool> F(const int &v0, const str &v1) const { return {}; }

  std::vector<bool> F(int v0, str v1, int v2) const { return {}; }

  std::vector<bool> F(int v0, str v1, int v2, str v3) const { return {}; }

  std::vector<bool> F(int v0, str v1, int v2, str v3, int v4) const { return {}; }

  std::vector<bool> F(int v0, str v1, int v2, str v3, int v4, str v5) const { return {}; }

  std::vector<bool> F(int v0, str v1, int v2, str v3, int v4, str v5, int v6) const { return {}; }

  std::vector<bool> F(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7) const { return {}; }

  std::vector<bool> F(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8) const { return {}; }

  std::vector<bool> F(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8, str v9) const {
    return {};
  }

  std::vector<bool> operator()() const { return {}; }

  std::vector<bool> operator()(const int &v0) const { return {}; }

  std::vector<bool> operator()(const int &v0, const str &v1) const { return {}; }

  std::vector<bool> operator()(int v0, str v1, int v2) const { return {}; }

  std::vector<bool> operator()(int v0, str v1, int v2, str v3) const { return {}; }

  std::vector<bool> operator()(int v0, str v1, int v2, str v3, int v4) const { return {}; }

  std::vector<bool> operator()(int v0, str v1, int v2, str v3, int v4, str v5) const { return {}; }

  std::vector<bool> operator()(int v0, str v1, int v2, str v3, int v4, str v5, int v6) const { return {}; }

  std::vector<bool> operator()(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7) const { return {}; }

  std::vector<bool> operator()(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8) const {
    return {};
  }

  std::vector<bool> operator()(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8, str v9) const {
    return {};
  }

#if 0
  std::vector<bool> &operator[]() { return member; }

  std::vector<bool> &operator[](const int &v0) { return member; }

  std::vector<bool> &operator[](const int &v0, const str &v1) { return member; }

  std::vector<bool> &operator[](int v0, str v1, int v2) { return member; }

  std::vector<bool> &operator[](int v0, str v1, int v2, str v3) { return member; }

  std::vector<bool> &operator[](int v0, str v1, int v2, str v3, int v4) { return member; }

  std::vector<bool> &operator[](int v0, str v1, int v2, str v3, int v4, str v5) { return member; }

  std::vector<bool> &operator[](int v0, str v1, int v2, str v3, int v4, str v5, int v6) { return member; }

  std::vector<bool> &operator[](int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7) { return member; }

  std::vector<bool> &operator[](int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8) {
    return member;
  }

  std::vector<bool> &operator[](int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8, str v9) {
    return member;
  }
#else
  const std::vector<bool> &operator[](const int &v0) const { return member; }

  std::vector<bool> &operator[](const int &v0) { return member; }
#endif

  static std::vector<bool> G() { return {}; }

  static std::vector<bool> G(const int &v0) { return {}; }

  static std::vector<bool> G(const int &v0, const str &v1) { return {}; }

  static std::vector<bool> G(int v0, str v1, int v2) { return {}; }

  static std::vector<bool> G(int v0, str v1, int v2, str v3) { return {}; }

  static std::vector<bool> G(int v0, str v1, int v2, str v3, int v4) { return {}; }

  static std::vector<bool> G(int v0, str v1, int v2, str v3, int v4, str v5) { return {}; }

  static std::vector<bool> G(int v0, str v1, int v2, str v3, int v4, str v5, int v6) { return {}; }

  static std::vector<bool> G(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7) { return {}; }

  static std::vector<bool> G(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8) { return {}; }

  static std::vector<bool> G(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8, str v9) {
    return {};
  }
};

std::vector<bool> F() {
  return {};
}

std::vector<bool> F(const int &v0) {
  return {};
}

std::vector<bool> F(const int &v0, const std::string &v1) {
  return {};
}

std::vector<bool> F(int v0, std::string v1, int v2) {
  return {};
}

std::vector<bool> F(int v0, std::string v1, int v2, std::string v3) {
  return {};
}

std::vector<bool> F(int v0, std::string v1, int v2, std::string v3, int v4) {
  return {};
}

std::vector<bool> F(int v0, std::string v1, int v2, std::string v3, int v4, std::string v5) {
  return {};
}

std::vector<bool> F(int v0, std::string v1, int v2, std::string v3, int v4, std::string v5, int v6) {
  return {};
}

std::vector<bool> F(int v0, std::string v1, int v2, std::string v3, int v4, std::string v5, int v6, std::string v7) {
  return {};
}

std::vector<bool>
F(int v0, std::string v1, int v2, std::string v3, int v4, std::string v5, int v6, std::string v7, int v8) {
  return {};
}

std::vector<bool> F(int v0,
                    std::string v1,
                    int v2,
                    std::string v3,
                    int v4,
                    std::string v5,
                    int v6,
                    std::string v7,
                    int v8,
                    std::string v9) {
  return {};
}

//
//
//
class ARIATestPython_Object {
public:
  std::vector<std::string> &name0() { return name_; }

public:
  ARIA_PROP_BEGIN(public, public, , std::vector<std::string>, name1);
  ARIA_PROP_FUNC(public, , ., clear);
  ARIA_PROP_END;

  ARIA_PROP_BEGIN(public, private, , std::vector<std::string>, name2);
  ARIA_PROP_END;

private:
  std::vector<std::string> name_ = {"Python です喵"}; // Test UTF-8.

  std::vector<std::string> ARIA_PROP_GETTER(name1)() const { return name_; }

  void ARIA_PROP_SETTER(name1)(const std::vector<std::string> &name) { name_ = name; }

  std::vector<std::string> ARIA_PROP_GETTER(name2)() const { return name_; }

  ARIA_PYTHON_TYPE_FRIEND;
};

//
//
//
class ARIATestPython_IntProperty {
public:
  ARIA_PROP(public, public, , int, value);

private:
  int value_ = 233;

  int ARIA_PROP_GETTER(value)() const { return value_; }

  void ARIA_PROP_SETTER(value)(const int &value) { value_ = value; }

  ARIA_PYTHON_TYPE_FRIEND;
};

} // namespace

//
//
//
//
//
// Define Python types.
ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_GrandParent);
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_Parent);
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_Child);
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_OverloadWithConst);
ARIA_PYTHON_TYPE_METHOD(const, value);
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_OverloadWithParameters);
ARIA_PYTHON_TYPE_METHOD(, value, int);
ARIA_PYTHON_TYPE_METHOD(, value, double);
ARIA_PYTHON_TYPE_METHOD(, value, int, double);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_ManyOverloads);
//
ARIA_PYTHON_TYPE_CONSTRUCTOR();
ARIA_PYTHON_TYPE_CONSTRUCTOR(const int &);
ARIA_PYTHON_TYPE_CONSTRUCTOR(const int &, const std::string &);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int, std::string, int);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int, std::string, int, std::string);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int, std::string, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int, std::string, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_CONSTRUCTOR(int, std::string, int, std::string, int, std::string, int, std::string, int, std::string);
//
ARIA_PYTHON_TYPE_METHOD(const, F);
ARIA_PYTHON_TYPE_METHOD(const, F, const int &);
ARIA_PYTHON_TYPE_METHOD(const, F, const int &, const std::string &);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(
    const, F, int, std::string, int, std::string, int, std::string, int, std::string, int, std::string);
//
ARIA_PYTHON_TYPE_OPERATOR_CALL(const);
ARIA_PYTHON_TYPE_OPERATOR_CALL(const, const int &);
ARIA_PYTHON_TYPE_OPERATOR_CALL(const, const int &, const std::string &);
ARIA_PYTHON_TYPE_OPERATOR_CALL(const, int, std::string, int);
ARIA_PYTHON_TYPE_OPERATOR_CALL(const, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_OPERATOR_CALL(const, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_OPERATOR_CALL(const, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_OPERATOR_CALL(const, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_OPERATOR_CALL(const, int, std::string, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_OPERATOR_CALL(const, int, std::string, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_OPERATOR_CALL(
    const, int, std::string, int, std::string, int, std::string, int, std::string, int, std::string);
//
#if 0
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, const int &);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, const int &, const std::string &);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, int, std::string, int);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, int, std::string, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, int, std::string, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(
    const, int, std::string, int, std::string, int, std::string, int, std::string, int, std::string);
#else
ARIA_PYTHON_TYPE_OPERATOR_ITEM(const, const int &);
ARIA_PYTHON_TYPE_OPERATOR_ITEM(, const int &);
#endif
//
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G, const int &);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G, const int &, const std::string &);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G, int, std::string, int);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G, int, std::string, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(G, int, std::string, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_STATIC_FUNCTION(
    G, int, std::string, int, std::string, int, std::string, int, std::string, int, std::string);
//
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F, const int &);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F, const int &, const std::string &);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F, int, std::string, int);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F, int, std::string, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(F, int, std::string, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(
    F, int, std::string, int, std::string, int, std::string, int, std::string, int, std::string);
//
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_Object);
ARIA_PYTHON_TYPE_METHOD(, name0);
ARIA_PYTHON_TYPE_PROPERTY(name1);
ARIA_PYTHON_TYPE_READONLY_PROPERTY(name2);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(decltype(std::declval<ARIATestPython_Object>().name1()));
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_METHOD(, clear);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(==, const std::vector<std::string> &);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(decltype(std::declval<ARIATestPython_Object>().name2()));
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(==, const std::vector<std::string> &);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_IntProperty);
ARIA_PYTHON_TYPE_PROPERTY(value);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(decltype(std::declval<ARIATestPython_IntProperty>().value()));
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_UNARY_OPERATOR(+);
ARIA_PYTHON_TYPE_UNARY_OPERATOR(-);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(==, decltype(std::declval<T>().value()));
ARIA_PYTHON_TYPE_BINARY_OPERATOR(+, decltype(std::declval<T>().value()));
ARIA_PYTHON_TYPE_BINARY_OPERATOR(-, decltype(std::declval<T>().value()));
ARIA_PYTHON_TYPE_BINARY_OPERATOR(*, decltype(std::declval<T>().value()));
ARIA_PYTHON_TYPE_BINARY_OPERATOR(/); // Test binary operators with self.
ARIA_PYTHON_ADD_TYPE(int);
ARIA_PYTHON_ADD_TYPE(double);
ARIA_PYTHON_ADD_TYPE(std::string);
ARIA_PYTHON_TYPE_END;

//
//
//
//
//
TEST(Python, Base) {
  Python::ScopedInterpreter guard{};

  Python::Module main = guard.Import("__main__");

  ARIA_PYTHON_ADD_TYPE(int, main);
  ARIA_PYTHON_ADD_TYPE(double, main);
  ARIA_PYTHON_ADD_TYPE(std::string, main);

  static_assert(main.HasType<int>());
  static_assert(main.HasType<double>());
  static_assert(main.HasType<std::string>());
  static_assert(main.HasType<std::tuple<int, double, std::string>>());
  EXPECT_FALSE(main.HasType<std::vector<int>>());
  EXPECT_TRUE((main.HasType<std::pair<std::string, std::vector<int>>>()));
  EXPECT_TRUE((main.HasType<std::tuple<int, double, std::string, std::vector<int>>>()));

  ARIA_PYTHON_ADD_TYPE(std::vector<int>, main);
  EXPECT_TRUE(main.HasType<std::vector<int>>());

  Python::Dict local{main};

  static_assert(std::is_same_v<std::decay_t<decltype("Hello")>, const char *>);

  local["a"] = "Hello";
  local["b"] = std::make_pair(1, 2);
  local["c"] = std::make_tuple(1, 2, 3);

  try {
    py::exec("assert a == 'Hello'\n"
             "assert b == (1, 2)\n"
             "assert c == (1, 2, 3)\n",
             py::globals(), local);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, Function) {
  Python::ScopedInterpreter guard{};

  // Get scope.
  Python::Module main = guard.Import("__main__");
  Python::Dict local{main};

  // Define functions.
  ARIA_PYTHON_ADD_FUNCTION(main, add, const std::vector<int> &, const std::vector<int> &);
  ARIA_PYTHON_ADD_FUNCTION(main, add, int, int);
  ARIA_PYTHON_ADD_FUNCTION(main, add, float, float);

  main.Def("add0", add0).Def("add1", [](const std::vector<int> &a, std::vector<int> &b) {
    size_t size = a.size();
    ARIA_ASSERT(size == b.size());

    std::vector<int> c(size);
    for (size_t i = 0; i < size; ++i)
      c[i] = a[i] + b[i];

    return c;
  });

  local["add2"] = py::cpp_function([](const std::vector<int> &a, std::vector<int> &b) {
    size_t size = a.size();
    ARIA_ASSERT(size == b.size());

    std::vector<int> c(size);
    for (size_t i = 0; i < size; ++i)
      c[i] = a[i] + b[i];

    return c;
  });

  // Define variables.
  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = {4, 6, 9};

  local["a"] = a;
  local["b"] = &b;

  // Execute.
  try {
    py::exec("c = add(a, b)\n"
             "c0 = add0(a, b)\n"
             "c1 = add1(a, b)\n"
             "c2 = add2(a, b)\n",
             py::globals(), local);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }

  auto c = local["c"].Cast<std::vector<int>>();
  EXPECT_EQ(c[0], 5);
  EXPECT_EQ(c[1], 8);
  EXPECT_EQ(c[2], 12);

  auto c0 = local["c0"].Cast<std::vector<int>>();
  EXPECT_EQ(c[0], c0[0]);
  EXPECT_EQ(c[1], c0[1]);
  EXPECT_EQ(c[2], c0[2]);

  auto c1 = local["c1"].Cast<std::vector<int>>();
  EXPECT_EQ(c[0], c1[0]);
  EXPECT_EQ(c[1], c1[1]);
  EXPECT_EQ(c[2], c1[2]);

  auto c2 = local["c2"].Cast<std::vector<int>>();
  EXPECT_EQ(c[0], c2[0]);
  EXPECT_EQ(c[1], c2[1]);
  EXPECT_EQ(c[2], c2[2]);
}

TEST(Python, Inheritance) {
  Python::ScopedInterpreter guard{};

  // Get scope.
  Python::Module main = guard.Import("__main__");
  Python::Dict local{main};

  // Define variables.
  ARIATestPython_Parent parent0;
  ARIATestPython_Child child0;
  ARIATestPython_Parent parent1;
  ARIATestPython_Child child1;
  std::shared_ptr<ARIATestPython_GrandParent> parent2 = std::make_shared<ARIATestPython_Parent>();
  std::unique_ptr<ARIATestPython_GrandParent> child2 = std::make_unique<ARIATestPython_Child>();

  local["parent0"] = parent0; // Pass by copy.
  local["child0"] = child0;
  local["parent1"] = &parent1; // Pass by pointer.
  local["child1"] = &child1;
  local["parent2"] = parent2.get(); // Pass by pointer.
  local["child2"] = child2.get();

  // Execute.
  try {
    py::exec("assert parent0.value() == 1\n"
             "assert child0.value() == 2\n"
             "assert parent1.value() == 1\n"
             "assert child1.value() == 2\n"
             "assert parent2.value() == 1\n"
             "assert child2.value() == 2\n",
             py::globals(), local);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, Const) {
  // TODO: Pybind11 always bypasses the `const` requirement.
  // TODO: For overloaded methods where `const` is the only difference,
  //       the earlier-defined one will be selected.

  // Bypass const.
  {
    Python::ScopedInterpreter guard{};

    // Get scope.
    Python::Module main = guard.Import("__main__");
    Python::Dict local{main};

    // Define variables.
    const ARIATestPython_Parent parent1;
    const ARIATestPython_Child child1;
    std::shared_ptr<const ARIATestPython_GrandParent> parent2 = std::make_shared<const ARIATestPython_Parent>();
    std::unique_ptr<const ARIATestPython_GrandParent> child2 = std::make_unique<const ARIATestPython_Child>();

    local["parent1"] = parent1; // Pass by copy.
    local["child1"] = child1;
    local["parent2"] = parent2.get(); // Pass by pointer.
    local["child2"] = child2.get();

    static_assert(std::is_same_v<decltype(parent2.get()), const ARIATestPython_GrandParent *>);
    static_assert(std::is_same_v<decltype(child2.get()), const ARIATestPython_GrandParent *>);

    // Execute.
    try {
      py::exec("assert parent1.value() == 1\n"
               "assert child1.value() == 2\n"
               "assert parent2.value() == 1\n"
               "assert child2.value() == 2\n",
               py::globals(), local);
    } catch (std::exception &e) {
      fmt::print("{}\n", e.what());
      EXPECT_FALSE(true);
    }
  }

  // Give const version higher priority.
  {
    Python::ScopedInterpreter guard{};

    // Get scope.
    Python::Module main = guard.Import("__main__");
    Python::Dict local{main};

    // Define variables.
    const ARIATestPython_OverloadWithConst overloadConst;
    ARIATestPython_OverloadWithConst overloadNonConst;

    local["overloadConst"] = overloadConst; // Pass by copy.
    local["overloadNonConst"] = overloadNonConst;

    // Execute.
    try {
      py::exec("assert overloadConst.value() == 0\n"
               "assert overloadNonConst.value() == 0\n",
               py::globals(), local);
    } catch (std::exception &e) {
      fmt::print("{}\n", e.what());
      EXPECT_FALSE(true);
    }
  }
}

TEST(Python, Overload) {
  Python::ScopedInterpreter guard{};

  // Get scope.
  Python::Module main = guard.Import("__main__");
  Python::Dict local{main};

  // Define variables.
  ARIATestPython_OverloadWithParameters overload;

  local["overload"] = overload;
  local["vector"] = std::vector<std::string>{"0"};

  EXPECT_TRUE(overload.value(0, 0.0) == std::vector<std::string>{"0"});

  // Execute.
  try {
    py::exec("assert overload.value(0) == 0\n"
             "assert overload.value(0.0) == '0'\n"
             "assert overload.value(0, 0.0) == vector\n",
             py::globals(), local);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, ManyOverloads) {
  Python::ScopedInterpreter guard{};

  // Get scope.
  Python::Module main = guard.Import("__main__");
  Python::Dict local{main};

  // Define variables.
  ARIATestPython_ManyOverloads manyOverloads;

  local["manyOverloads"] = manyOverloads;
  local["vector"] = std::vector<bool>{};
  local["vectorFilled"] = std::vector<bool>{true, false, true};

  // Execute.
  try {
    py::exec("assert manyOverloads.F() == vector\n"
             "assert manyOverloads.F(0) == vector\n"
             "assert manyOverloads.F(0, '1') == vector\n"
             "assert manyOverloads.F(0, '1', 2) == vector\n"
             "assert manyOverloads.F(0, '1', 2, '3') == vector\n"
             "assert manyOverloads.F(0, '1', 2, '3', 4) == vector\n"
             "assert manyOverloads.F(0, '1', 2, '3', 4, '5') == vector\n"
             "assert manyOverloads.F(0, '1', 2, '3', 4, '5', 6) == vector\n"
             "assert manyOverloads.F(0, '1', 2, '3', 4, '5', 6, '7') == vector\n"
             "assert manyOverloads.F(0, '1', 2, '3', 4, '5', 6, '7', 8) == vector\n"
             "assert manyOverloads.F(0, '1', 2, '3', 4, '5', 6, '7', 8, '9') == vector\n"
             "\n"
             "assert manyOverloads() == vector\n"
             "assert manyOverloads(0) == vector\n"
             "assert manyOverloads(0, '1') == vector\n"
             "assert manyOverloads(0, '1', 2) == vector\n"
             "assert manyOverloads(0, '1', 2, '3') == vector\n"
             "assert manyOverloads(0, '1', 2, '3', 4) == vector\n"
             "assert manyOverloads(0, '1', 2, '3', 4, '5') == vector\n"
             "assert manyOverloads(0, '1', 2, '3', 4, '5', 6) == vector\n"
             "assert manyOverloads(0, '1', 2, '3', 4, '5', 6, '7') == vector\n"
             "assert manyOverloads(0, '1', 2, '3', 4, '5', 6, '7', 8) == vector\n"
             "assert manyOverloads(0, '1', 2, '3', 4, '5', 6, '7', 8, '9') == vector\n"
             "\n"
             "assert manyOverloads[0] == vector\n"
             "manyOverloads[0] = vectorFilled\n"
             "assert manyOverloads[0] == vectorFilled\n"
             "manyOverloads[0] = vector\n"
             "assert manyOverloads[0] == vector\n"
             "\n"
             "assert ARIATestPython_ManyOverloads.G() == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0) == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0, '1') == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0, '1', 2) == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0, '1', 2, '3') == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0, '1', 2, '3', 4) == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0, '1', 2, '3', 4, '5') == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0, '1', 2, '3', 4, '5', 6) == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0, '1', 2, '3', 4, '5', 6, '7') == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0, '1', 2, '3', 4, '5', 6, '7', 8) == vector\n"
             "assert ARIATestPython_ManyOverloads.G(0, '1', 2, '3', 4, '5', 6, '7', 8, '9') == vector\n"
             "\n"
             "assert F() == vector\n"
             "assert F(0) == vector\n"
             "assert F(0, '1') == vector\n"
             "assert F(0, '1', 2) == vector\n"
             "assert F(0, '1', 2, '3') == vector\n"
             "assert F(0, '1', 2, '3', 4) == vector\n"
             "assert F(0, '1', 2, '3', 4, '5') == vector\n"
             "assert F(0, '1', 2, '3', 4, '5', 6) == vector\n"
             "assert F(0, '1', 2, '3', 4, '5', 6, '7') == vector\n"
             "assert F(0, '1', 2, '3', 4, '5', 6, '7', 8) == vector\n"
             "assert F(0, '1', 2, '3', 4, '5', 6, '7', 8, '9') == vector\n"
             "\n"
             "a0 = ARIATestPython_ManyOverloads()\n"
             "a1 = ARIATestPython_ManyOverloads(0)\n"
             "a2 = ARIATestPython_ManyOverloads(0, '1')\n"
             "a3 = ARIATestPython_ManyOverloads(0, '1', 2)\n"
             "a4 = ARIATestPython_ManyOverloads(0, '1', 2, '3')\n"
             "a5 = ARIATestPython_ManyOverloads(0, '1', 2, '3', 4)\n"
             "a6 = ARIATestPython_ManyOverloads(0, '1', 2, '3', 4, '5')\n"
             "a7 = ARIATestPython_ManyOverloads(0, '1', 2, '3', 4, '5', 6)\n"
             "a8 = ARIATestPython_ManyOverloads(0, '1', 2, '3', 4, '5', 6, '7')\n"
             "a9 = ARIATestPython_ManyOverloads(0, '1', 2, '3', 4, '5', 6, '7', 8)\n"
             "a10 = ARIATestPython_ManyOverloads(0, '1', 2, '3', 4, '5', 6, '7', 8, '9')\n"
             "\n"
             "assert a10.F() == vector\n"
             "assert a9.F(0) == vector\n"
             "assert a8.F(0, '1') == vector\n"
             "assert a7.F(0, '1', 2) == vector\n"
             "assert a6.F(0, '1', 2, '3') == vector\n"
             "assert a5.F(0, '1', 2, '3', 4) == vector\n"
             "assert a4.F(0, '1', 2, '3', 4, '5') == vector\n"
             "assert a3.F(0, '1', 2, '3', 4, '5', 6) == vector\n"
             "assert a2.F(0, '1', 2, '3', 4, '5', 6, '7') == vector\n"
             "assert a1.F(0, '1', 2, '3', 4, '5', 6, '7', 8) == vector\n"
             "assert a0.F(0, '1', 2, '3', 4, '5', 6, '7', 8, '9') == vector\n",
             py::globals(), local);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, Properties) {
  Python::ScopedInterpreter guard{};

  // Get scope.
  Python::Module main = guard.Import("__main__");
  Python::Dict local{main};

  // Define variables.
  std::vector<std::string> nameCase0 = {"Python です喵"};
  std::vector<std::string> nameCase1 = {"Python 喵です"};
  std::vector<std::string> nameCase2 = {};
  ARIATestPython_Object obj;

  local["nameCase0"] = nameCase0;
  local["nameCase1"] = nameCase1;
  local["nameCase2"] = nameCase2;
  local["obj"] = obj;

  // Execute.
  try {
    py::exec("assert obj.name0() == obj.name0()\n"
             "assert obj.name1 == obj.name1\n"
             "assert obj.name0() == obj.name1\n"
             "assert obj.name1 == obj.name0()\n"
             "assert obj.name0() == nameCase0\n"
             "assert obj.name1 == nameCase0\n"
             "assert nameCase0 == obj.name0()\n"
             "assert nameCase0 == obj.name1\n"
             "assert obj.name0() == obj.name0()\n"
             "assert obj.name2 == obj.name2\n"
             "assert obj.name0() == obj.name2\n"
             "assert obj.name2 == obj.name0()\n"
             "assert obj.name0() == nameCase0\n"
             "assert obj.name2 == nameCase0\n"
             "assert nameCase0 == obj.name0()\n"
             "assert nameCase0 == obj.name2\n"
             "\n"
             "obj.name1 = nameCase1\n" // Test setter with `operator=`.
             "assert obj.name0() == obj.name0()\n"
             "assert obj.name1 == obj.name1\n"
             "assert obj.name0() == obj.name1\n"
             "assert obj.name1 == obj.name0()\n"
             "assert obj.name0() == nameCase1\n"
             "assert obj.name1 == nameCase1\n"
             "assert nameCase1 == obj.name0()\n"
             "assert nameCase1 == obj.name1\n"
             "assert obj.name0() == obj.name0()\n"
             "assert obj.name2 == obj.name2\n"
             "assert obj.name0() == obj.name2\n"
             "assert obj.name2 == obj.name0()\n"
             "assert obj.name0() == nameCase1\n"
             "assert obj.name2 == nameCase1\n"
             "assert nameCase1 == obj.name0()\n"
             "assert nameCase1 == obj.name2\n"
             "\n"
             "obj.name1.clear()\n" // Test setter with `clear()`.
             "assert obj.name0() == obj.name0()\n"
             "assert obj.name1 == obj.name1\n"
             "assert obj.name0() == obj.name1\n"
             "assert obj.name1 == obj.name0()\n"
             "assert obj.name0() == nameCase2\n"
             "assert obj.name1 == nameCase2\n"
             "assert nameCase2 == obj.name0()\n"
             "assert nameCase2 == obj.name1\n"
             "assert obj.name0() == obj.name0()\n"
             "assert obj.name2 == obj.name2\n"
             "assert obj.name0() == obj.name2\n"
             "assert obj.name2 == obj.name0()\n"
             "assert obj.name0() == nameCase2\n"
             "assert obj.name2 == nameCase2\n"
             "assert nameCase2 == obj.name0()\n"
             "assert nameCase2 == obj.name2\n",
             py::globals(), local);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, Operators) {
  Python::ScopedInterpreter guard{};

  // Get scope.
  Python::Module main = guard.Import("__main__");
  Python::Dict local{main};

  // Define variables.
  ARIATestPython_IntProperty intP;

  local["intP"] = intP;

  // Execute.
  try {
    py::exec("assert intP.value == intP.value\n"
             "assert intP.value == 233\n"
             "assert 233 == intP.value\n"
             "\n"
             "intP.value = -1\n"
             "assert intP.value == intP.value\n"
             "assert intP.value == -1\n"
             "assert -1 == intP.value\n"
             "\n"
             "intP.value = intP.value + -5\n"
             "assert intP.value == intP.value\n"
             "assert intP.value == -6\n"
             "assert -6 == intP.value\n"
             "\n"
             "intP.value = intP.value - -7\n"
             "assert intP.value == intP.value\n"
             "assert intP.value == 1\n"
             "assert 1 == intP.value\n"
             "\n"
             "intP.value = intP.value * -3\n"
             "assert intP.value == intP.value\n"
             "assert intP.value == -3\n"
             "assert -3 == intP.value\n"
             "\n"
             "intP.value = +intP.value\n"
             "assert intP.value == intP.value\n"
             "assert intP.value == -3\n"
             "assert -3 == intP.value\n"
             "\n"
             "intP.value = -intP.value\n"
             "assert intP.value == intP.value\n"
             "assert intP.value == +3\n"
             "assert +3 == intP.value\n"
             "\n"
             "intP.value = intP.value / intP.value\n"
             "assert intP.value == intP.value\n"
             "assert intP.value == 1\n"
             "assert 1 == intP.value\n",
             py::globals(), local);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

} // namespace ARIA
