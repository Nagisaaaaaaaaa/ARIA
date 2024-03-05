#pragma once

#include "ARIA/Property.h"

#include <pybind11/embed.h>
#include <pybind11/operators.h>

namespace ARIA {

namespace py = pybind11;

//
//
//
template <typename T>
void DefinePythonType(const py::module_ &module);

//
//
//
#define __ARIA_PYTHON_TYPE_FRIEND                                                                                      \
                                                                                                                       \
  template <typename TUVW>                                                                                             \
  friend void ::ARIA::DefinePythonType(const py::module_ &module)

//
//
//
#define __ARIA_PYTHON_TYPE_BEGIN(TYPE)                                                                                 \
                                                                                                                       \
  template <>                                                                                                          \
  void DefinePythonType<TYPE>(const py::module_ &module) {                                                             \
    using Type = TYPE;                                                                                                 \
                                                                                                                       \
    py::class_<Type> cls(module, #TYPE)

//
//
//
// clang-format off
#define __ARIA_PYTHON_TYPE_METHOD_PARAMS2(SPECIFIERS, NAME)                                                            \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  ))                                                                                                                   \
  (Type::*)() SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS3(SPECIFIERS, NAME, T0)                                                        \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>()))                                                                                                 \
  (Type::*)(T0) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS4(SPECIFIERS, NAME, T0, T1)                                                    \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (Type::*)(T0, T1) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS5(SPECIFIERS, NAME, T0, T1, T2)                                                \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (Type::*)(T0, T1, T2) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS6(SPECIFIERS, NAME, T0, T1, T2, T3)                                            \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (Type::*)(T0, T1, T2, T3) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS7(SPECIFIERS, NAME, T0, T1, T2, T3, T4)                                        \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>()))                 \
  (Type::*)(T0, T1, T2, T3, T4) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS8(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5)                                    \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>()))                                                                                                 \
  (Type::*)(T0, T1, T2, T3, T4, T5) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS9(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6)                                \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>()))                                                                             \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS10(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6, T7)                           \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>()))                                                         \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6, T7) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS11(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8)                       \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>()))                                     \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6, T7, T8) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS12(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)                   \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>(), std::declval<T9>()))                 \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) SPECIFIERS>(&Type::NAME))
// clang-format on

#define __ARIA_PYTHON_TYPE_METHOD(...)                                                                                 \
  __ARIA_EXPAND(__ARIA_EXPAND(ARIA_CONCAT(__ARIA_PYTHON_TYPE_METHOD_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

//
//
//
#define __ARIA_PYTHON_TYPE_PROPERTY(NAME)                                                                              \
                                                                                                                       \
  static_assert(property::detail::PropertyType<decltype(std::declval<Type>().NAME())>,                                 \
                "The given type to be defined in Python should be an ARIA property type");                             \
  cls.def_property(#NAME, static_cast<decltype(std::declval<Type>().NAME()) (Type::*)()>(&Type::NAME),                 \
                   static_cast<void (Type::*)(const decltype(std::declval<Type>().ARIA_PROP_IMPL(NAME)()) &)>(         \
                       &Type::ARIA_PROP_IMPL(NAME)))

//
//
//
#define __ARIA_PYTHON_TYPE_READONLY_PROPERTY(NAME)                                                                     \
                                                                                                                       \
  static_assert(property::detail::PropertyType<decltype(std::declval<Type>().NAME())>,                                 \
                "The given type to be defined in Python should be an ARIA property type");                             \
  cls.def_property_readonly(#NAME, static_cast<decltype(std::declval<Type>().NAME()) (Type::*)()>(&Type::NAME))

//
//
//
#define __ARIA_PYTHON_TYPE_UNARY_OPERATOR(OPERATOR) cls.def(decltype(OPERATOR py::self)())

//
//
//
#define __ARIA_PYTHON_TYPE_BINARY_OPERATOR_PARAMS1(OPERATOR) cls.def(decltype(py::self OPERATOR py::self)());

#define __ARIA_PYTHON_TYPE_BINARY_OPERATOR_PARAMS2(OPERATOR, OTHERS)                                                   \
                                                                                                                       \
  cls.def(decltype(py::self OPERATOR py::self)());                                                                     \
  cls.def(decltype(py::self OPERATOR std::declval<OTHERS>())());                                                       \
  cls.def(decltype(std::declval<OTHERS>() OPERATOR py::self)())

#define __ARIA_PYTHON_TYPE_BINARY_OPERATOR(...)                                                                        \
  __ARIA_EXPAND(                                                                                                       \
      __ARIA_EXPAND(ARIA_CONCAT(__ARIA_PYTHON_TYPE_BINARY_OPERATOR_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

//
//
//
#define __ARIA_PYTHON_TYPE_END                                                                                         \
  }                                                                                                                    \
  namespace py = pybind11

//
//
//
#define __ARIA_ADD_PYTHON_TYPE(TYPE, MODULE) ::ARIA::DefinePythonType<TYPE>(main)

//
//
//
//
//
class module_item_accessor {
public:
  explicit module_item_accessor(py::module_ module, py::detail::item_accessor accessor)
      : module_(std::move(module)), accessor_(std::move(accessor)) {}

  ARIA_COPY_MOVE_ABILITY(module_item_accessor, default, default);

public:
  template <typename T>
  void operator=(T &&value) {
    // TODO: Calls ARIA_ADD_PYTHON_TYPE and recursively define types.

    accessor_ = std::forward<T>(value);
  }

private:
  py::module_ module_;
  py::detail::item_accessor accessor_;
};

class module_local {
public:
  explicit module_local(py::module module) : module_(std::move(module)) {}

  ARIA_COPY_MOVE_ABILITY(module_local, delete, delete);

public:
private:
  py::module_ module_;
};

} // namespace ARIA
