#pragma once

#include "ARIA/Property.h"

#include <pybind11/embed.h>
#include <pybind11/operators.h>

#include <list>

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
//
//
//
//
class Module {
public:
  [[nodiscard]] bool HasType(const std::string &name) { return types_->contains(name); }

public:
  ARIA_COPY_MOVE_ABILITY(Module, default, default);

private:
  friend class ScopedInterpreter;

  // The constructor is only allowed to be called by `ScopedInterpreter::Import()`.
  Module(py::module_ module, std::unordered_set<std::string> &types) : module_(std::move(module)), types_(&types) {}

  py::module_ module_;

  // Have to use pointer instead of reference here to allow copying and moving.
  std::unordered_set<std::string> *types_{};
};

//
//
//
class ScopedInterpreter {
public:
  explicit ScopedInterpreter(bool initSignalHandlers = true,
                             int argc = 0,
                             const char *const *argv = nullptr,
                             bool addProgramDirToPath = true)
      : interpreter_(initSignalHandlers, argc, argv, addProgramDirToPath) {}

#if PY_VERSION_HEX >= PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX
  explicit ScopedInterpreter(PyConfig *config,
                             int argc = 0,
                             const char *const *argv = nullptr,
                             bool addProgramDirToPath = true)
      : interpreter_(config, argc, argv, addProgramDirToPath) {}
#endif

  ARIA_COPY_MOVE_ABILITY(ScopedInterpreter, delete, delete);

public:
  [[nodiscard]] Module Import(const char *name) {
    py::module_ module = py::module_::import(name);

    // First check whether the module has been imported sometime earlier.
    for (auto &types : moduleTypes_)
      if (types.first == name)
        return {std::move(module), types.second};

    // If the module has not been imported, create an instance for it.
    auto &types = moduleTypes_.emplace_back(std::string{name}, std::unordered_set<std::string>{});

    return {std::move(module), types.second};
  }

private:
  py::scoped_interpreter interpreter_;

  // A "singleton" dictionary containing all the types defined by all the imported modules.
  // The layout looks like this:
  //   [ "__main__"   : { "Vec3i", "Vec3f", ... },
  //     "someModule" : { "Object", "Transform", ... },
  //     ... ]
  std::list<std::pair<std::string, std::unordered_set<std::string>>> moduleTypes_;
};

//
//
//
namespace python::detail {

class ItemAccessor {
public:
  ItemAccessor(Module module, py::detail::item_accessor accessor)
      : module_(std::move(module)), accessor_(std::move(accessor)) {}

  ARIA_COPY_MOVE_ABILITY(ItemAccessor, default, default);

public:
  template <typename T>
  void operator=(T &&value) {
    // TODO: Calls `ARIA_ADD_PYTHON_TYPE` and recursively define types for `module_`.

    accessor_ = std::forward<T>(value);
  }

private:
  Module module_;
  py::detail::item_accessor accessor_;
};

} // namespace python::detail

//
//
//
class Dict {
public:
  explicit Dict(Module module) : module_(std::move(module)) {}

  ARIA_COPY_MOVE_ABILITY(Dict, default, default);

  operator py::dict() { return dict_; }

public:
  python::detail::ItemAccessor operator[](py::handle key) const { return {module_, dict_[key]}; }

  python::detail::ItemAccessor operator[](py::object &&key) const { return {module_, dict_[std::move(key)]}; }

  python::detail::ItemAccessor operator[](const char *key) const { return {module_, dict_[pybind11::str(key)]}; }

private:
  Module module_;
  py::dict dict_;
};

} // namespace ARIA
