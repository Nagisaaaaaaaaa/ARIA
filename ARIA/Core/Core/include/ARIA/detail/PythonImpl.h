#pragma once

#include "ARIA/Property.h"
#include "ARIA/TypeArray.h"

#include <pybind11/embed.h>
#include <pybind11/operators.h>

#include <list>

namespace ARIA {

namespace py = pybind11;

//
//
//
namespace python::detail {

template <typename T>
struct is_std_pair_or_std_tuple : std::false_type {};

template <typename... Args>
struct is_std_pair_or_std_tuple<std::pair<Args...>> : std::true_type {};

template <typename... Args>
struct is_std_pair_or_std_tuple<std::tuple<Args...>> : std::true_type {};

template <typename T>
inline constexpr bool is_std_pair_or_std_tuple_v = is_std_pair_or_std_tuple<T>::value;

//
//
//
template <typename T>
struct is_method {
  static constexpr bool value = false;
};

template <type_array::detail::NonArrayType Ret,
          type_array::detail::NonArrayType T,
          type_array::detail::NonArrayType... Args>
struct is_method<Ret (T::*)(Args...)> {
  static constexpr bool value = true;

  using return_type = Ret;
  using arguments_types = MakeTypeArray<Args...>;
  using return_and_arguments_types = MakeTypeArray<return_type, arguments_types>;
};

template <type_array::detail::NonArrayType Ret,
          type_array::detail::NonArrayType T,
          type_array::detail::NonArrayType... Args>
struct is_method<Ret (T::*)(Args...) const> {
  static constexpr bool value = true;

  using return_type = Ret;
  using arguments_types = MakeTypeArray<Args...>;
  using return_and_arguments_types = MakeTypeArray<return_type, arguments_types>;
};

template <typename T>
static constexpr bool is_method_v = is_method<T>::value;

template <typename T>
concept method = is_method_v<T>;

//
//
//
template <typename T>
consteval bool is_python_builtin_type() {
  using TDecayed = std::decay_t<T>;

  // Return true if `TDecayed` is a Python-builtin type, which is
  // fundamental, `const char*`, `std::string`, `std::pair`, or `std::tuple`,
  // because these types have been implicitly handled by pybind11.
  //! Note that it is possible for `std::pair` and `std::tuple` to
  //! contain unhandled types, for example, `std::pair<int, std::vector<int>>`, where
  //! `std::vector<int>` is unhandled.
  //! There's no way to perfectly address this problem.
  if constexpr (std::is_fundamental_v<TDecayed> || std::is_same_v<TDecayed, const char *> ||
                std::is_same_v<TDecayed, std::string> || python::detail::is_std_pair_or_std_tuple_v<TDecayed>) {
    //! `T` is not allowed to be a non-const reference type, as explained below.
    static_assert(!(!std::is_const_v<std::remove_reference_t<T>> && std::is_reference_v<T>),
                  "It is dangerous to take non-const references for Python-builtin types because"
                  "these types are immutable in Python codes thus will result in undefined behaviors");

    return true;
  }

  return false;
}

} // namespace python::detail

//
//
//
class Module;

template <typename T>
struct __ARIAPython_RecursivelyDefinePythonType {};

//
//
//
class Module {
public:
  template <typename T>
  [[nodiscard]] constexpr bool HasType() const {
    using TDecayed = std::decay_t<T>;

    //! Non-const references to Python-builtin types have already been checked here.
    if constexpr (python::detail::is_python_builtin_type<T>())
      return true;

    // Check whether the unordered set contains the hash code.
    return types_->contains(typeid(TDecayed).hash_code());
  }

public:
  ARIA_COPY_MOVE_ABILITY(Module, default, default);

  operator py::module() const { return module_; }

private:
  friend class ScopedInterpreter;

  // The constructor is only allowed to be called by `ScopedInterpreter::Import()`.
  Module(py::module_ module, std::unordered_set<size_t> &types) : module_(std::move(module)), types_(&types) {}

  py::module_ module_;

  // Have to use pointer instead of reference here to allow copying and moving.
  std::unordered_set<size_t> *types_{};

  template <typename TUVW>
  friend struct ::ARIA::__ARIAPython_RecursivelyDefinePythonType;
};

//
//
//
template <python::detail::method TMethod>
struct __ARIAPython_RecursivelyDefinePythonType<TMethod> {
  void operator()(const Module &module) {
    ForEach<typename python::detail::is_method<TMethod>::return_and_arguments_types>([&]<typename T>() {
      using TDecayed = std::decay_t<T>;

      __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<TDecayed>>>()(module);
    });
  }
};

template <typename T>
  requires(python::detail::is_python_builtin_type<T>())
struct __ARIAPython_RecursivelyDefinePythonType<T> {
  void operator()(const Module &module) {
    // Do nothing for Python-builtin types.
  }
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
    auto &types = moduleTypes_.emplace_back(std::string{name}, std::unordered_set<size_t>{});

    return {std::move(module), types.second};
  }

private:
  py::scoped_interpreter interpreter_;

  // A "singleton" dictionary containing all the types defined by all the imported modules.
  // The layout looks like this:
  //   [ "__main__"   : { typeid(...).hash_code(), ... },
  //     "someModule" : { typeid(...).hash_code(), ... },
  //     ... ]
  std::list<std::pair<std::string, std::unordered_set<size_t>>> moduleTypes_;
};

//
//
//
namespace python::detail {

// TODO: Directly wrapping a `py::item_accessor` will result in runtime error, why?
template <typename Arg>
class ItemAccessor {
public:
  ItemAccessor(Module module, py::dict dict, Arg arg)
      : module_(std::move(module)), dict_(std::move(dict)), arg_(std::move(arg)) {}

  ARIA_COPY_MOVE_ABILITY(ItemAccessor, default, default);

public:
  template <typename T>
  void operator=(T &&value) {
    using TDecayed = std::decay_t<T>;

    if constexpr (!std::is_same_v<TDecayed, py::cpp_function>)
      __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<TDecayed>>>()(module_);

    dict_[arg_] = std::forward<T>(value);
  }

  template <typename T>
  decltype(auto) Cast() const {
    return dict_[arg_].template cast<T>();
  }

  template <typename T>
  decltype(auto) Cast() {
    return dict_[arg_].template cast<T>();
  }

private:
  Module module_;
  py::dict dict_;
  Arg arg_;
};

} // namespace python::detail

//
//
//
class Dict {
public:
  explicit Dict(Module module) : module_(std::move(module)) {}

  ARIA_COPY_MOVE_ABILITY(Dict, default, default);

  operator py::dict() const { return dict_; }

public:
  auto operator[](py::handle key) const { return python::detail::ItemAccessor{module_, dict_, key}; }

  auto operator[](py::object &&key) const { return python::detail::ItemAccessor{module_, dict_, std::move(key)}; }

  auto operator[](const char *key) const { return python::detail::ItemAccessor{module_, dict_, key}; }

private:
  Module module_;
  py::dict dict_;
};

//
//
//
//
//
//
//
//
//
#define __ARIA_PYTHON_TYPE_FRIEND                                                                                      \
                                                                                                                       \
  template <typename TUVW>                                                                                             \
  friend struct ::ARIA::__ARIAPython_RecursivelyDefinePythonType;

//
//
//
#define __ARIA_PYTHON_TYPE_BEGIN(TYPE)                                                                                 \
                                                                                                                       \
  template <>                                                                                                          \
  struct __ARIAPython_RecursivelyDefinePythonType<TYPE> {                                                              \
    void operator()(const Module &module) {                                                                            \
      using Type = TYPE;                                                                                               \
                                                                                                                       \
      static_assert(std::is_same_v<Type, std::decay_t<Type>>,                                                          \
                    "The given type to be defined in Python should be a decayed type");                                \
                                                                                                                       \
      static_assert(!std::is_const_v<Type>, "The given type to be defined in Python should not be a const type "       \
                                            "because `const` has no effect in Python");                                \
      static_assert(!std::is_pointer_v<Type>, "The given type to be defined in Python should not be a pointer type "   \
                                              "because pointer types are automatically handled");                      \
                                                                                                                       \
      static_assert(!python::detail::is_python_builtin_type<Type>(),                                                   \
                    "The given type to be defined in Python should not be a Python-builtin type");                     \
      static_assert(!python::detail::method<Type>,                                                                     \
                    "The given type to be defined in Python should not be a method type");                             \
                                                                                                                       \
      /* Return if this type has already been defined in this module. */                                               \
      if (module.HasType<Type>())                                                                                      \
        return;                                                                                                        \
                                                                                                                       \
      /* If this type has not been defined in this module, mark it as defined. */                                      \
      module.types_->insert(typeid(Type).hash_code());                                                                 \
                                                                                                                       \
      /* Define this type in this module. */                                                                           \
      py::class_<Type> cls(module, #TYPE)

//
//
//
// clang-format off
#define __ARIA_PYTHON_TYPE_METHOD_PARAMS2(SPECIFIERS, NAME)                                                            \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  ))                                                                                                                   \
  (Type::*)() SPECIFIERS>()(module);                                                                                     \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  ))                                                                                                                   \
  (Type::*)() SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS3(SPECIFIERS, NAME, T0)                                                        \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>()))                                                                                                 \
  (Type::*)(T0) SPECIFIERS>()(module);                                                                                   \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>()))                                                                                                 \
  (Type::*)(T0) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS4(SPECIFIERS, NAME, T0, T1)                                                    \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (Type::*)(T0, T1) SPECIFIERS>()(module);                                                                               \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (Type::*)(T0, T1) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS5(SPECIFIERS, NAME, T0, T1, T2)                                                \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (Type::*)(T0, T1, T2) SPECIFIERS>()(module);                                                                           \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (Type::*)(T0, T1, T2) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS6(SPECIFIERS, NAME, T0, T1, T2, T3)                                            \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (Type::*)(T0, T1, T2, T3) SPECIFIERS>()(module);                                                                       \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (Type::*)(T0, T1, T2, T3) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS7(SPECIFIERS, NAME, T0, T1, T2, T3, T4)                                        \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>()))                 \
  (Type::*)(T0, T1, T2, T3, T4) SPECIFIERS>()(module);                                                                   \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>()))                 \
  (Type::*)(T0, T1, T2, T3, T4) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS8(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5)                                    \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>()))                                                                                                 \
  (Type::*)(T0, T1, T2, T3, T4, T5) SPECIFIERS>()(module);                                                               \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>()))                                                                                                 \
  (Type::*)(T0, T1, T2, T3, T4, T5) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS9(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6)                                \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>()))                                                                             \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6) SPECIFIERS>()(module);                                                           \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>()))                                                                             \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS10(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6, T7)                           \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>()))                                                         \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6, T7) SPECIFIERS>()(module);                                                       \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>()))                                                         \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6, T7) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS11(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8)                       \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>()))                                     \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6, T7, T8) SPECIFIERS>()(module);                                                   \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>()))                                     \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6, T7, T8) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS12(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)                   \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS Type>().NAME(                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>(), std::declval<T9>()))                 \
  (Type::*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) SPECIFIERS>()(module);                                               \
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
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<Type>().NAME()) (Type::*)()>()(module);               \
  __ARIAPython_RecursivelyDefinePythonType<void (Type::*)(                                                             \
      const decltype(std::declval<Type>().ARIA_PROP_IMPL(NAME)()) &)>()(module);                                       \
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
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<Type>().NAME()) (Type::*)()>()(module);               \
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
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<OTHERS>>>>()(        \
      module);                                                                                                         \
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
  }                                                                                                                    \
  ;                                                                                                                    \
  namespace py = pybind11

//
//
//
#define __ARIA_ADD_PYTHON_TYPE(TYPE, MODULE) ::ARIA::__ARIAPython_RecursivelyDefinePythonType<TYPE>()(MODULE)

//
//
//
//
//
//
//
//
//
// TODO: Support template defining and add `PythonSTL.h`.
__ARIA_PYTHON_TYPE_BEGIN(std::vector<bool>);
__ARIA_PYTHON_TYPE_METHOD(, clear);
__ARIA_PYTHON_TYPE_BINARY_OPERATOR(==);
__ARIA_PYTHON_TYPE_END;

__ARIA_PYTHON_TYPE_BEGIN(std::vector<int>);
__ARIA_PYTHON_TYPE_METHOD(, clear);
__ARIA_PYTHON_TYPE_BINARY_OPERATOR(==);
__ARIA_PYTHON_TYPE_END;

__ARIA_PYTHON_TYPE_BEGIN(std::vector<std::string>);
__ARIA_PYTHON_TYPE_METHOD(, clear);
__ARIA_PYTHON_TYPE_BINARY_OPERATOR(==);
__ARIA_PYTHON_TYPE_END;

} // namespace ARIA
