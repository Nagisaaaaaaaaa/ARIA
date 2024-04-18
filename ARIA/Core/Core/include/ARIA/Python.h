#pragma once

/// \file
/// \brief Python is embedded as a script language in ARIA.
/// You are able to easily run Python scripts in a 100% C++ environment.
///
/// `pybind11` is used as raw APIs for basic C++-Python interoperability, while
/// some APIs are wrapped for easier usage, for example, C++ types can be
/// automatically and recursively defined in Python, with minimal efforts.
///
/// Make sure you are familiar with Python and `pybind11` before continue, see
/// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html#evaluating-python-expressions-from-strings-and-files
//
//
//
//
//
#include "ARIA/detail/PythonImpl.h"

namespace ARIA {

/// \brief While namespace `Python` contains all the wrapped APIs,
/// namespace `py` contains all the raw APIs, for example, `py::exec`.
///
/// Wrapped and raw APIs can be used together safely in ARIA
/// (just like CUDA runtime API and CUDA API wrapper), but
/// we recommend using the wrapped APIs as much as you can.
namespace py = python::detail::py;

//
//
//
/// \brief A RAII class for the Python interpreter.
/// This class is a wrapper for `py::scoped_interpreter`.
///
/// \example ```cpp
/// Python::ScopedInterpreter interpreter{};
///
/// py::exec("print('Hello Python!')\n");
/// ```
///
/// \details This class should be used instead of `py::scoped_interpreter` because
/// it contains more contexts about Python modules, for example,
/// which C++ types have currently been defined for each module.
/// These contexts are helpful for automatically and recursively defining types.
namespace Python {
using python::detail::ScopedInterpreter;
}

//
//
//
/// \brief A Python module.
/// This class is a wrapper for `py::module`.
///
/// \example ```cpp
/// Python::ScopedInterpreter interpreter{};
/// Python::Module main = interpreter.Import("__main__");
///
/// static_assert(main.HasType<int>());
/// EXPECT_FALSE(main.HasType<std::vector<int>>());
/// ```
///
/// \details This class should be used instead of `py::module_` because
/// it contains more contexts, for example,
/// which C++ types have currently been defined for this module.
/// These contexts are helpful for automatically and recursively defining types.
namespace Python {
using python::detail::Module;
}

//
//
//
/// \brief A Python dictionary, which can be used to
/// represent global or local variables.
///
/// \example ```cpp
/// Python::ScopedInterpreter interpreter{};
/// Python::Module main = interpreter.Import("__main__");
/// Python::Dict local{main};
///
/// // Define a function which adds 2 `std::vector<int>` and returns the sum.
/// local["add"] = py::cpp_function([](const std::vector<int> &a, const std::vector<int> &b) {
///   size_t size = a.size();
///   ARIA_ASSERT(size == b.size());
///
///   std::vector<int> c(size);
///   for (size_t i = 0; i < size; ++i)
///     c[i] = a[i] + b[i];
///
///   return c;
/// });
///
/// // Define 2 variables.
/// std::vector<int> a = {1, 2, 3};
/// std::vector<int> b = {4, 6, 9};
///
/// local["a"] = a;  // `a` is passed to Python by value.
/// local["b"] = &b; // `b` is passed to Python by reference,
///
/// // Call the Python interpreter and computes a + b.
/// py::exec("c = add(a, b)\n",
///          py::globals(), // Global variables: set to default.
///          local);        // Local variables: set to `local`.
///
/// // Get the computation result from Python.
/// auto c = local["c"].Cast<std::vector<int>>();
/// EXPECT_EQ(c[0], 5);
/// EXPECT_EQ(c[1], 8);
/// EXPECT_EQ(c[2], 12);
/// ```
///
/// \details If you are familiar with libraries such as pybind, you may have
/// noticed that `std::vector<int>` is not explicitly defined in this example.
/// That is the magic provided by ARIA `ScopedInterpreter`, `Module`, and `Dict`.
/// Types are automatically and recursively defined at `= a` and `= &b`.
namespace Python {
using python::detail::Dict;
}

//
//
//
//
//
/// \brief The wrapped APIs allow automatically and recursively defining Python types,
/// as introduced in the comments of `Dict`.
/// To make it work, we should define how a C++ class looks like in Python,
/// for example, which methods and properties should be defined.
/// This is done within the region of `ARIA_PYTHON_TYPE_BEGIN` and `ARIA_PYTHON_TYPE_END`.
///
/// \example ```cpp
/// class Object { ... };
///
/// ARIA_PYTHON_TYPE_BEGIN(Object);
/// ...
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_BEGIN(type) __ARIA_PYTHON_TYPE_BEGIN(type)

//
//
//
/// \brief A template version of `ARIA_PYTHON_TYPE_BEGIN`.
///
/// \example ```cpp
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(std::vector);
/// ...
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(template_) __ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(template_)

//
//
//
/// \brief We usually want to define private methods to Python,
/// especially while defining ARIA properties, whose
/// getters and setters are always private.
///
/// This macro should be used to give the Python APIs
/// accessibility to private methods.
///
/// \example ```
/// class Object {
/// private:
///   ...
///   ARIA_PYTHON_TYPE_FRIEND;
/// };
/// ```
#define ARIA_PYTHON_TYPE_FRIEND __ARIA_PYTHON_TYPE_FRIEND

//
//
//
/// \brief Define a constructor for the given type or template.
///
/// \example ```cpp
/// class Object {
/// public:
///   Object() { ... }
///   explicit Object(const std::string& name) { ... }
/// };
///
/// template <typename T, auto size>
/// class Vec {
/// public:
///   using value_type = T;
///
///   Vec() { ... }
///   explicit Vec(const value_type& v) { ... }
/// }
///
/// ARIA_PYTHON_TYPE_BEGIN(Object);
/// ARIA_PYTHON_TYPE_CONSTRUCTOR(); // The zero-argument constructor.
/// ARIA_PYTHON_TYPE_CONSTRUCTOR(const std::string&);
/// ARIA_PYTHON_TYPE_END;
///
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(Vec);
/// ARIA_PYTHON_TYPE_CONSTRUCTOR(); // The zero-argument constructor.
/// // Here, `T` is a place holder equals to the instantiated template type, that is, `Vec<T, size>`.
/// ARIA_PYTHON_TYPE_CONSTRUCTOR(const T::value_type&);
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_CONSTRUCTOR /* (parameters...) */ __ARIA_PYTHON_TYPE_CONSTRUCTOR

//
//
//
/// \brief Define a method for the given type or template.
///
/// \example ```cpp
/// class Object {
/// public:
///   const std::string& name() const { ... }
/// };
///
/// template <typename T, auto size>
/// class Vec {
/// public:
///   void Normalize() { ... }
///   T Dot(const Vec& others) const { ... }
/// };
///
/// ARIA_PYTHON_TYPE_BEGIN(Object);
/// ARIA_PYTHON_TYPE_METHOD(const, name);
/// ARIA_PYTHON_TYPE_END;
///
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(Vec);
/// ARIA_PYTHON_TYPE_METHOD(, Normalize);
/// // Here, `T` is a place holder equals to the instantiated template type, that is, `Vec<T, size>`.
/// ARIA_PYTHON_TYPE_METHOD(const, Dot, const T&);
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_METHOD /* (specifiers, name, parameters...) */ __ARIA_PYTHON_TYPE_METHOD

//
//
//
/// \brief Define a static member function for the given type or template.
///
/// \example ```cpp
/// template <typename T, auto size>
/// class Vec {
/// public:
///   static Vec Zero() { ... }
///   static T Dot(const Vec& x, const Vec& y) { ... }
/// };
///
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(Vec);
/// ARIA_PYTHON_TYPE_STATIC_FUNCTION(Zero);
/// // Here, `T` is a place holder equals to the instantiated template type, that is, `Vec<T, size>`.
/// ARIA_PYTHON_TYPE_STATIC_FUNCTION(Dot, const T&, const T&);
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_STATIC_FUNCTION /* (name, parameters...) */ __ARIA_PYTHON_TYPE_STATIC_FUNCTION

//
//
//
/// \brief Define an ARIA property for the given type or template.
///
/// \example ```cpp
/// class Object {
/// public:
///   ARIA_PROP(public, public, , std::string, name);
/// private:
///   std::string ARIA_PROP_IMPL(name)() const { ... }
///   void ARIA_PROP_IMPL(name)(const std::string &name) { ... }
/// };
///
/// ARIA_PYTHON_TYPE_BEGIN(Object);
/// ARIA_PYTHON_TYPE_PROPERTY(name);
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_PROPERTY(name) __ARIA_PYTHON_TYPE_PROPERTY(name)

//
//
//
/// \brief Define a readonly ARIA property for the given type or template.
///
/// \example ```cpp
/// class Object {
/// public:
///   ARIA_PROP(public, private, , std::string, name);
/// private:
///   std::string ARIA_PROP_IMPL(name)() const { ... }
/// };
///
/// ARIA_PYTHON_TYPE_BEGIN(Object);
/// ARIA_PYTHON_TYPE_READONLY_PROPERTY(name);
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_READONLY_PROPERTY(name) __ARIA_PYTHON_TYPE_READONLY_PROPERTY(name)

//
//
//
/// \brief Define an unary operator for the given type or template.
///
/// \example ```cpp
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(Vec);
/// ARIA_PYTHON_TYPE_UNARY_OPERATOR(+); // Enables `b = +a` in Python.
/// ARIA_PYTHON_TYPE_UNARY_OPERATOR(-); // Enables `b = -a` in Python.
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_UNARY_OPERATOR(op) __ARIA_PYTHON_TYPE_UNARY_OPERATOR(op)

//
//
//
/// \brief Define a binary operator for the given type or template.
///
/// \example ```cpp
/// template <typename T, auto size>
/// class Vec {
/// public:
///   using value_type = T;
///   ...
/// };
///
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(Vec);
/// ARIA_PYTHON_TYPE_BINARY_OPERATOR(==); // Only enables `a == b` in Python, `a` and `b` should have the same type.
/// ARIA_PYTHON_TYPE_BINARY_OPERATOR(*, T::value_type); // Enables `c = a * b`, `c = a * 2`, and `c = 2 * a` in Python.
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_BINARY_OPERATOR /* (op) or (op, others) */ __ARIA_PYTHON_TYPE_BINARY_OPERATOR

//
//
//
/// \brief Define an external function for the given type or template.
///
/// \example ```cpp
/// template <typename T, auto size>
/// class Vec { ... };
///
/// template <typename T, auto size>
/// Vec abs(const Vec& x) { ... }
///
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(Vec);
/// // Here, `T` is a place holder equals to the instantiated template type, that is, `Vec<T, size>`.
/// ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(abs, const T&);
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION /* (name, parameters...) */ __ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION

//
//
//
/// \brief Finish the definition of how a C++ class looks like in Python.
/// This macro should be used together with `ARIA_PYTHON_TYPE_BEGIN`.
///
/// \see ARIA_PYTHON_TYPE_BEGIN
#define ARIA_PYTHON_TYPE_END __ARIA_PYTHON_TYPE_END

//
//
//
/// \brief Manually define the given type in Python.
/// This macro can be used within the region of
/// `ARIA_PYTHON_TYPE_BEGIN` and `ARIA_PYTHON_TYPE_END`, or
/// be called like a function.
///
/// \example ```cpp
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(Vec);
/// // By adding the following 2 lines, template `Vec` depends on
/// // both `float16` and `bfloat16`.
/// // When any instantiated template type `Vec<T, size>` is defined in Python,
/// // both `float16` and `bfloat16` will be automatically and recursively defined.
/// ARIA_PYTHON_ADD_TYPE(float16);
/// ARIA_PYTHON_ADD_TYPE(bfloat16);
/// ARIA_PYTHON_TYPE_END;
///
/// Python::ScopedInterpreter interpreter{};
/// Python::Module main = interpreter.Import("__main__");
/// // Define `std::vector<int>` for module `main`.
/// ARIA_PYTHON_ADD_TYPE(std::vector<int>, main);
/// EXPECT_TRUE(main.HasType<std::vector<int>>());
/// ```
#define ARIA_PYTHON_ADD_TYPE /* (type) or (type, module) */ __ARIA_PYTHON_ADD_TYPE

//
//
//
/// \brief Manually define the given function in Python.
///
/// \example ```cpp
/// Python::ScopedInterpreter interpreter{};
/// Python::Module main = interpreter.Import("__main__");
///
/// // Define the following `abs` for module `main`.
/// ARIA_PYTHON_ADD_FUNCTION(main, abs, int);
/// ARIA_PYTHON_ADD_FUNCTION(main, abs, int64);
/// ARIA_PYTHON_ADD_FUNCTION(main, abs, float);
/// ARIA_PYTHON_ADD_FUNCTION(main, abs, double);
/// ```
#define ARIA_PYTHON_ADD_FUNCTION /* (module, name, parameters...) */ __ARIA_PYTHON_ADD_FUNCTION

} // namespace ARIA
