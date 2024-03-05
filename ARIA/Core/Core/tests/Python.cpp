#include "ARIA/Python.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct GrandParent {
  virtual ~GrandParent() = default;

  virtual int value() = 0;
};

struct Parent : public GrandParent {
  virtual ~Parent() = default;

  int value() override { return 1; }
};

struct Child final : public Parent {
  virtual ~Child() = default;

  int value() final { return 2; }
};

//
//
//
struct OverloadWithConst {
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

  std::vector<bool> F() const { return {}; }

  std::vector<bool> F(int v0) const { return {}; }

  std::vector<bool> F(int v0, str v1) const { return {}; }

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
};

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

  std::vector<std::string> ARIA_PROP_IMPL(name1)() const { return name_; }

  void ARIA_PROP_IMPL(name1)(const std::vector<std::string> &name) { name_ = name; }

  std::vector<std::string> ARIA_PROP_IMPL(name2)() const { return name_; }

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

  int ARIA_PROP_IMPL(value)() const { return value_; }

  void ARIA_PROP_IMPL(value)(const int &value) { value_ = value; }

  ARIA_PYTHON_TYPE_FRIEND;
};

} // namespace

//
//
//
//
//
// Define Python types.
ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_OverloadWithParameters);
ARIA_PYTHON_TYPE_METHOD(, value, int);
ARIA_PYTHON_TYPE_METHOD(, value, double);
ARIA_PYTHON_TYPE_METHOD(, value, int, double);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_ManyOverloads);
ARIA_PYTHON_TYPE_METHOD(const, F);
ARIA_PYTHON_TYPE_METHOD(const, F, int);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F, int, std::string, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(
    const, F, int, std::string, int, std::string, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_Object);
ARIA_PYTHON_TYPE_METHOD(, name0);
ARIA_PYTHON_TYPE_PROPERTY(name1);
ARIA_PYTHON_TYPE_READONLY_PROPERTY(name2);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(decltype(std::declval<ARIATestPython_Object>().name1()));
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_METHOD(, clear);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(==, std::vector<std::string>);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(decltype(std::declval<ARIATestPython_Object>().name2()));
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(==, std::vector<std::string>);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(ARIATestPython_IntProperty);
ARIA_PYTHON_TYPE_PROPERTY(value);
ARIA_PYTHON_TYPE_END;

ARIA_PYTHON_TYPE_BEGIN(decltype(std::declval<ARIATestPython_IntProperty>().value()));
ARIA_PYTHON_TYPE_METHOD(, value);
ARIA_PYTHON_TYPE_UNARY_OPERATOR(+);
ARIA_PYTHON_TYPE_UNARY_OPERATOR(-);
ARIA_PYTHON_TYPE_BINARY_OPERATOR(==, decltype(std::declval<ARIATestPython_IntProperty>().value().value()));
ARIA_PYTHON_TYPE_BINARY_OPERATOR(+, decltype(std::declval<ARIATestPython_IntProperty>().value().value()));
ARIA_PYTHON_TYPE_BINARY_OPERATOR(-, decltype(std::declval<ARIATestPython_IntProperty>().value().value()));
ARIA_PYTHON_TYPE_BINARY_OPERATOR(*, decltype(std::declval<ARIATestPython_IntProperty>().value().value()));
ARIA_PYTHON_TYPE_BINARY_OPERATOR(
    /, decltype(std::declval<ARIATestPython_IntProperty>().value().value())); // Test binary operators with self.
ARIA_PYTHON_TYPE_END;

//
//
//
//
//
TEST(Python, Base) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<std::vector<int>>(main, "std::vector<int>");

  // Define functions.
  locals["add"] = py::cpp_function([](const std::vector<int> &a, std::vector<int> &b) {
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

  locals["a"] = py::cast(a, py::return_value_policy::reference);
  locals["b"] = py::cast(b, py::return_value_policy::reference);

  // Execute.
  try {
    py::exec("c = add(a, b)\n"
             "\n",
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }

  auto c = locals["c"].cast<std::vector<int>>();
  EXPECT_EQ(c[0], 5);
  EXPECT_EQ(c[1], 8);
  EXPECT_EQ(c[2], 12);
}

TEST(Python, Inheritance) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<GrandParent>(main, "GrandParent").def("value", &GrandParent::value);
  py::class_<Parent, GrandParent>(main, "Parent").def("value", &Parent::value);
  py::class_<Child, Parent>(main, "Child").def("value", &Child::value);

  // Define variables.
  Parent parent0;
  Child child0;
  Parent parent1;
  Child child1;
  std::shared_ptr<GrandParent> parent2 = std::make_shared<Parent>();
  std::unique_ptr<GrandParent> child2 = std::make_unique<Child>();

  locals["parent0"] = parent0; // Pass by copy.
  locals["child0"] = child0;
  locals["parent1"] = py::cast(parent1, py::return_value_policy::reference); // Pass by reference.
  locals["child1"] = py::cast(child1, py::return_value_policy::reference);
  locals["parent2"] = parent2.get(); // Pass by pointer.
  locals["child2"] = child2.get();

  // Execute.
  try {
    py::exec("assert parent0.value() == 1\n"
             "assert child0.value() == 2\n"
             "assert parent1.value() == 1\n"
             "assert child1.value() == 2\n"
             "assert parent2.value() == 1\n"
             "assert child2.value() == 2\n",
             py::globals(), locals);
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
    py::scoped_interpreter guard{};

    // Get scope.
    py::object main = py::module_::import("__main__");
    py::dict locals;

    // Define types.
    py::class_<GrandParent>(main, "GrandParent").def("value", &GrandParent::value);
    py::class_<Parent, GrandParent>(main, "Parent").def("value", &Parent::value);
    py::class_<Child, Parent>(main, "Child").def("value", &Child::value);

    // Define variables.
    const Parent parent1;
    const Child child1;
    std::shared_ptr<const GrandParent> parent2 = std::make_shared<const Parent>();
    std::unique_ptr<const GrandParent> child2 = std::make_unique<const Child>();

    locals["parent1"] = py::cast(parent1, py::return_value_policy::reference); // Pass by reference.
    locals["child1"] = py::cast(child1, py::return_value_policy::reference);
    locals["parent2"] = parent2.get(); // Pass by pointer.
    locals["child2"] = child2.get();

    static_assert(std::is_same_v<decltype(parent2.get()), const GrandParent *>);
    static_assert(std::is_same_v<decltype(child2.get()), const GrandParent *>);

    // Execute.
    try {
      py::exec("assert parent1.value() == 1\n"
               "assert child1.value() == 2\n"
               "assert parent2.value() == 1\n"
               "assert child2.value() == 2\n",
               py::globals(), locals);
    } catch (std::exception &e) {
      fmt::print("{}\n", e.what());
      EXPECT_FALSE(true);
    }
  }

  // Give const version higher priority.
  {
    py::scoped_interpreter guard{};

    // Get scope.
    py::object main = py::module_::import("__main__");
    py::dict locals;

    // Define types.
    py::class_<OverloadWithConst>(main, "OverloadWithConst")
        .def("value",
             static_cast<decltype(std::declval<const OverloadWithConst>().value()) (OverloadWithConst::*)() const>(
                 &OverloadWithConst::value))
        .def("value", static_cast<decltype(std::declval<OverloadWithConst>().value()) (OverloadWithConst::*)()>(
                          &OverloadWithConst::value));

    // Define variables.
    const OverloadWithConst overloadConst;
    OverloadWithConst overloadNonConst;

    locals["overloadConst"] = py::cast(overloadConst, py::return_value_policy::reference); // Pass by reference.
    locals["overloadNonConst"] = py::cast(overloadNonConst, py::return_value_policy::reference);

    // Execute.
    try {
      py::exec("assert overloadConst.value() == 0\n"
               "assert overloadNonConst.value() == 0\n",
               py::globals(), locals);
    } catch (std::exception &e) {
      fmt::print("{}\n", e.what());
      EXPECT_FALSE(true);
    }
  }

  // Give non-const version higher priority.
  {
    py::scoped_interpreter guard{};

    // Get scope.
    py::object main = py::module_::import("__main__");
    py::dict locals;

    // Define types.
    py::class_<OverloadWithConst>(main, "OverloadWithConst")
        .def("value", static_cast<decltype(std::declval<OverloadWithConst>().value()) (OverloadWithConst::*)()>(
                          &OverloadWithConst::value))
        .def("value",
             static_cast<decltype(std::declval<const OverloadWithConst>().value()) (OverloadWithConst::*)() const>(
                 &OverloadWithConst::value));

    // Define variables.
    const OverloadWithConst overloadConst;
    OverloadWithConst overloadNonConst;

    locals["overloadConst"] = py::cast(overloadConst, py::return_value_policy::reference); // Pass by reference.
    locals["overloadNonConst"] = py::cast(overloadNonConst, py::return_value_policy::reference);

    // Execute.
    try {
      py::exec("assert overloadConst.value() == 1\n"
               "assert overloadNonConst.value() == 1\n",
               py::globals(), locals);
    } catch (std::exception &e) {
      fmt::print("{}\n", e.what());
      EXPECT_FALSE(true);
    }
  }
}

TEST(Python, Overload) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<std::vector<std::string>>(main, "std::vector<std::string>").def(py::self == py::self);

  ARIA_ADD_PYTHON_TYPE(ARIATestPython_OverloadWithParameters, main);

  // Define variables.
  ARIATestPython_OverloadWithParameters overload;

  locals["overload"] = py::cast(overload, py::return_value_policy::reference);
  locals["vector"] = std::vector<std::string>{"0"};

  EXPECT_TRUE(overload.value(0, 0.0) == std::vector<std::string>{"0"});

  // Execute.
  try {
    py::exec("assert overload.value(0) == 0\n"
             "assert overload.value(0.0) == '0'\n"
             "assert overload.value(0, 0.0) == vector\n",
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, ManyOverloads) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<std::vector<bool>>(main, "std::vector<bool>").def(py::self == py::self);

  ARIA_ADD_PYTHON_TYPE(ARIATestPython_ManyOverloads, main);

  // Define variables.
  ARIATestPython_ManyOverloads manyOverloads;

  locals["manyOverloads"] = py::cast(manyOverloads, py::return_value_policy::reference);
  locals["vector"] = std::vector<bool>{};

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
             "assert manyOverloads.F(0, '1', 2, '3', 4, '5', 6, '7', 8, '9') == vector\n",
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, Properties) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<std::vector<std::string>>(main, "std::vector<std::string>")
      .def(py::self == py::self)
      .def("clear", &std::vector<std::string>::clear);

  ARIA_ADD_PYTHON_TYPE(ARIATestPython_Object, main);
  ARIA_ADD_PYTHON_TYPE(decltype(std::declval<ARIATestPython_Object>().name1()), main);
  ARIA_ADD_PYTHON_TYPE(decltype(std::declval<ARIATestPython_Object>().name2()), main);

  // Define variables.
  std::vector<std::string> nameCase0 = {"Python です喵"};
  std::vector<std::string> nameCase1 = {"Python 喵です"};
  std::vector<std::string> nameCase2 = {};
  ARIATestPython_Object obj;

  locals["nameCase0"] = py::cast(nameCase0, py::return_value_policy::reference);
  locals["nameCase1"] = py::cast(nameCase1, py::return_value_policy::reference);
  locals["nameCase2"] = py::cast(nameCase2, py::return_value_policy::reference);
  locals["obj"] = py::cast(obj, py::return_value_policy::reference);

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
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, Operators) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  ARIA_ADD_PYTHON_TYPE(ARIATestPython_IntProperty, main);
  ARIA_ADD_PYTHON_TYPE(decltype(std::declval<ARIATestPython_IntProperty>().value()), main);

  // Define variables.
  ARIATestPython_IntProperty intP;

  locals["intP"] = py::cast(intP, py::return_value_policy::reference);

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
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

//
//
//
//
//
TEST(Python, WarppedScopedInterpreterAndModule) {
  ScopedInterpreter guard{};

  Module main = guard.Import("__main__");
  EXPECT_FALSE(main.HasType("std::vector<int>"));
}

} // namespace ARIA
