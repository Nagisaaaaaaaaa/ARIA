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

  std::vector<bool> F0() const { return {}; }

  std::vector<bool> F1(int v0) const { return {}; }

  std::vector<bool> F2(int v0, str v1) const { return {}; }

  std::vector<bool> F3(int v0, str v1, int v2) const { return {}; }

  std::vector<bool> F4(int v0, str v1, int v2, str v3) const { return {}; }

  std::vector<bool> F5(int v0, str v1, int v2, str v3, int v4) const { return {}; }

  std::vector<bool> F6(int v0, str v1, int v2, str v3, int v4, str v5) const { return {}; }

  std::vector<bool> F7(int v0, str v1, int v2, str v3, int v4, str v5, int v6) const { return {}; }

  std::vector<bool> F8(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7) const { return {}; }

  std::vector<bool> F9(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8) const { return {}; }

  std::vector<bool> F10(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8, str v9) const {
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

  ARIA_PROP_BEGIN(public, public, , std::vector<std::string>, name2);
  ARIA_PROP_FUNC(public, , ., clear);
  ARIA_PROP_END;

private:
  std::vector<std::string> name_ = {"Python です喵"}; // Test UTF-8.

  std::vector<std::string> ARIA_PROP_IMPL(name1)() const { return name_; }

  void ARIA_PROP_IMPL(name1)(const std::vector<std::string> &name) { name_ = name; }

  std::vector<std::string> ARIA_PROP_IMPL(name2)() const { return name_; }

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
ARIA_PYTHON_TYPE_METHOD(const, F0);
ARIA_PYTHON_TYPE_METHOD(const, F1, int);
ARIA_PYTHON_TYPE_METHOD(const, F2, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F3, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(const, F4, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F5, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(const, F6, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F7, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(const, F8, int, std::string, int, std::string, int, std::string, int, std::string);
ARIA_PYTHON_TYPE_METHOD(const, F9, int, std::string, int, std::string, int, std::string, int, std::string, int);
ARIA_PYTHON_TYPE_METHOD(
    const, F10, int, std::string, int, std::string, int, std::string, int, std::string, int, std::string);
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
  // TODO: Test this.
}

TEST(Python, Operators) {
  // TODO: Test this.
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
             "assert obj.name0() == nameCase0\n" // Test getter.
             "assert obj.name1 == nameCase0\n"
             "assert nameCase0 == obj.name0()\n"
             "assert nameCase0 == obj.name1\n"
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
             "\n"
             "obj.name1.clear()\n" // Test setter with `clear()`.
             "assert obj.name0() == obj.name0()\n"
             "assert obj.name1 == obj.name1\n"
             "assert obj.name0() == obj.name1\n"
             "assert obj.name1 == obj.name0()\n"
             "assert obj.name0() == nameCase2\n"
             "assert obj.name1 == nameCase2\n"
             "assert nameCase2 == obj.name0()\n"
             "assert nameCase2 == obj.name1\n",
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, ReadonlyProperties) {
  // TODO: Test this.
}

//
//
//
// TODO: 谁负责调用 ARIA_ADD_PYTHON_TYPE 呢？
//       用户不应该显示地调用这个函数，否则很容易弄出 bug 来
//       我们需要封装 module 这个概念
//       然后需要定义 local 这个概念，local 至少要隶属于某个 module 才是合法的
//       接着，当我调用 local[...] = ... 时
//       这个 operator= 才会真正调用 ARIA_ADD_PYTHON_TYPE
//       它需要做这么几件事情：
//         1. 检查是否是内置类型，例如 int, float, std::string 等
//         2. 如果是内置类型，不允许传入引用，否则会危险
//         3. 如果不是内置类型，检查 decayed 这个类型是否已经被 ARIA_PYTHON_TYPE_BEGIN 定义
//         4. 如果被定义，检查是否已经被这个 module add 过
//         5. 如果是 reference，包一层 capsule
//         6. 最后将变量传给 Python local
//
// TODO: 别忘了，注册需要是递归的：
//       对于任何一个要注册的 class，我要递归注册：
//         1. 对于所有 method，递归注册所有的输入参数和输出参数类型
//         2. 对于所有 property，递归注册它们
//         3. 小心死循环，eg: obj.parent.parent....
//         4. 递归注册本质上就是递归地调用 ARIA_ADD_PYTHON_TYPE, 仅此而已
//       因此，我们还需要“模板”版本的 ARIA_ADD_PYTHON_TYPE，或者定义一系列 prefabs
//
// TODO: 那么谁负责调用 local[...] = ... 呢？
//       可以定义一个类似叫做 PythonMonoBehavior 的类
//       这个类在初始化的时候，会调用 local[transform] = object.transform() 等操作
//       因此，每个 PythonMonoBehavior 有自己的 local，但是共享同一个 module
//       这里，local 不能是共享的，因为会有变量名冲突
//       module 需要是共享的，因为都位于 ARIA
//       类似，globals 需要是共享的，这个概念也需要被定义出来
//
// TODO: 但是这并不见得是个好事情，我们更希望 Python 被放在 Core 里面，而不是 scene
//       更好的做法可能是把它设计得更底层一些，例如定义这些 class：
//         1. Python::Interpreter: 封装了对 Python 字符串脚本或者文件的调用
//         2. Python::Generator: 可暂停的 Python 字符串脚本或者文件
//                它的实现可以很简单，因为我们有解释器，可以为 Python 定义关键字
//                比如 yield 就可以是一个关键字，表示暂停
//                只需要写一个 parser 就完事了
//         3. 那很显然地，可以做到任何 coroutine 能做到的事情
//            但具体应该设计成什么样我还没想清楚，困困喵～

} // namespace ARIA
