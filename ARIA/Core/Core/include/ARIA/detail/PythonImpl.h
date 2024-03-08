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

// Whether the given type is `std::pair` or `std::tuple`.
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
// Whether the given type is a method (non-static member function) type.
template <typename T>
struct is_method {
  static constexpr bool value = false;
};

// Specialization for non-const methods.
template <type_array::detail::NonArrayType Ret,
          type_array::detail::NonArrayType T,
          type_array::detail::NonArrayType... Args>
struct is_method<Ret (T::*)(Args...)> {
  static constexpr bool value = true;

  using return_type = Ret;
  using arguments_types = MakeTypeArray<Args...>;
  using return_and_arguments_types = MakeTypeArray<return_type, arguments_types>;
};

// Specialization for const methods.
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
// Whether the given type is a Python-builtin type.
//
// Python-builtin types are taken special attention because
// their hash codes are not contained in the unordered set of `Module`.
template <typename T>
[[nodiscard]] consteval bool is_python_builtin_type() {
  using TUndecorated = std::remove_const_t<std::remove_pointer_t<std::decay_t<T>>>;

  // Return true if `TUndecorated` is a Python-builtin type, which is
  // fundamental, `std::string`, `std::pair`, or `std::tuple`,
  // because these types have been implicitly handled by pybind11.
  //! Note that it is possible for `std::pair` and `std::tuple` to
  //! contain unhandled types, for example, `std::pair<int, std::vector<int>>`, where
  //! `std::vector<int>` is unhandled.
  //! There's no way to perfectly address this problem.
  if constexpr (std::is_fundamental_v<TUndecorated> || std::is_same_v<TUndecorated, std::string> ||
                python::detail::is_std_pair_or_std_tuple_v<TUndecorated>) {
    //! `T` is not allowed to be a non-const reference type, as explained below.
    static_assert(!(!std::is_const_v<std::remove_reference_t<T>> && std::is_reference_v<std::remove_const_t<T>>),
                  "It is dangerous to take non-const references for Python-builtin types because"
                  "these types are immutable in Python codes thus will result in weird behaviors");

    return true;
  }

  return false;
}

} // namespace python::detail

//
//
//
//
//
// `operator()` of this class is called to
// define the C++ type `T` in the given Python module.
template <typename T>
struct __ARIAPython_RecursivelyDefinePythonType {
  // void operator()(const Module &module) {}
};

//
//
//
class Module {
public:
  /// \brief `Module` is implemented with reference counter thus
  /// supports both copy and move.
  ARIA_COPY_MOVE_ABILITY(Module, default, default);

  /// \brief `Module` can be implicitly cast to `py::module`.
  operator py::module() const { return module_; }

public:
  /// \brief Whether the given type has been defined in this module.
  ///
  /// \example ```cpp
  /// bool has = module.HasType<std::vector<int>>();
  /// ```
  template <typename T>
  [[nodiscard]] constexpr bool HasType() const {
    using TUndecorated = std::remove_const_t<std::remove_pointer_t<std::decay_t<T>>>;

    //! Non-const references to Python-builtin types have already been checked here.
    if constexpr (python::detail::is_python_builtin_type<T>())
      return true;

    // Check whether the unordered set contains the hash code.
    return types_->contains(typeid(TUndecorated).hash_code());
  }

private:
  // The constructor is only allowed to be called by `ScopedInterpreter::Import()`.
  friend class ScopedInterpreter;

  Module(py::module_ module, std::unordered_set<size_t> &types) : module_(std::move(module)), types_(&types) {}

  py::module_ module_;

  // The `std::unordered_set` is owned by the interpreter, like a singleton.
  // It contains hash codes of all the types which have been defined in this module.
  // Note, Python-builtin types are excluded.
  //
  // Have to use pointer instead of reference here to allow copying and moving.
  std::unordered_set<size_t> *types_{};

  // We will insert types to the `std::unordered_set` within `__ARIAPython_RecursivelyDefinePythonType`, so
  // `friend` is required here.
  template <typename TUVW>
  friend struct ::ARIA::__ARIAPython_RecursivelyDefinePythonType;
};

//
//
//
// Specialization for method types.
template <python::detail::method TMethod>
struct __ARIAPython_RecursivelyDefinePythonType<TMethod> {
  void operator()(const Module &module) {
    // Define a method in Python will define its return type and all its arguments types.
    ForEach<typename python::detail::is_method<TMethod>::return_and_arguments_types>([&]<typename T>() {
      using TUndecorated = std::remove_const_t<std::remove_pointer_t<std::decay_t<T>>>;

      __ARIAPython_RecursivelyDefinePythonType<TUndecorated>()(module);
    });
  }
};

// Specialization for Python-builtin types.
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
  /// \see py::scoped_interpreter
  explicit ScopedInterpreter(bool initSignalHandlers = true,
                             int argc = 0,
                             const char *const *argv = nullptr,
                             bool addProgramDirToPath = true)
      : interpreter_(initSignalHandlers, argc, argv, addProgramDirToPath) {}

#if PY_VERSION_HEX >= PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX
  /// \see py::scoped_interpreter
  explicit ScopedInterpreter(PyConfig *config,
                             int argc = 0,
                             const char *const *argv = nullptr,
                             bool addProgramDirToPath = true)
      : interpreter_(config, argc, argv, addProgramDirToPath) {}
#endif

  ARIA_COPY_MOVE_ABILITY(ScopedInterpreter, delete, delete);

public:
  /// \brief Import the Python module by name.
  ///
  /// \see py::module_::import
  [[nodiscard]] Module Import(const char *name) {
    py::module_ module = py::module_::import(name);

    // First check whether the module has been imported sometime earlier.
    for (auto &types : moduleTypes_)
      if (types.first == name)
        return {std::move(module), types.second}; // Return the wrapped module.

    // If the module has not been imported, create an instance for it.
    auto &types = moduleTypes_.emplace_back(std::string{name}, std::unordered_set<size_t>{});

    return {std::move(module), types.second}; // Return the wrapped module.
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

template <typename Arg>
class ItemAccessor {
public:
  ItemAccessor(Module module, py::dict dict, Arg arg)
      : module_(std::move(module)), dict_(std::move(dict)), arg_(std::move(arg)) {}

  ARIA_COPY_MOVE_ABILITY(ItemAccessor, default, default);

public:
  /// \see py::item_accessor::operator=
  template <typename T>
  void operator=(T &&value) {
    using TUndecorated = std::remove_const_t<std::remove_pointer_t<std::decay_t<T>>>;

    // If `TUndecorated` is not `py::cpp_function`, recursively define it in Python.
    if constexpr (!std::is_same_v<TUndecorated, py::cpp_function>)
      __ARIAPython_RecursivelyDefinePythonType<TUndecorated>()(module_);

    dict_[arg_] = std::forward<T>(value);
  }

  /// \see py::item_accessor::cast
  template <typename T>
  [[nodiscard]] decltype(auto) Cast() const {
    return dict_[arg_].template cast<T>();
  }

  /// \see py::item_accessor::cast
  template <typename T>
  [[nodiscard]] decltype(auto) Cast() {
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

  /// \brief `Dict` is implemented with reference counter thus
  /// supports both copy and move.
  ARIA_COPY_MOVE_ABILITY(Dict, default, default);

  /// \brief `Dict` can be implicitly cast to `py::dict`.
  operator py::dict() const { return dict_; }

public:
  /// \see py::dict::operator[]
  auto operator[](py::handle key) const { return python::detail::ItemAccessor{module_, dict_, key}; }

  /// \see py::dict::operator[]
  auto operator[](py::object &&key) const { return python::detail::ItemAccessor{module_, dict_, std::move(key)}; }

  /// \see py::dict::operator[]
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
      using T = TYPE;                                                                                                  \
                                                                                                                       \
      static_assert(std::is_same_v<T, std::decay_t<T>>,                                                                \
                    "The given type to be defined in Python should be a decayed type");                                \
                                                                                                                       \
      static_assert(!std::is_const_v<T>, "The given type to be defined in Python should not be a const type "          \
                                         "because `const` has no effect in Python");                                   \
      static_assert(!std::is_pointer_v<T>, "The given type to be defined in Python should not be a pointer type "      \
                                           "because pointer types are automatically handled");                         \
                                                                                                                       \
      static_assert(!python::detail::is_python_builtin_type<T>(),                                                      \
                    "The given type to be defined in Python should not be a Python-builtin type");                     \
      static_assert(!python::detail::method<T>, "The given type to be defined in Python should not be a method type"); \
                                                                                                                       \
      /* Return if this type has already been defined in this module. */                                               \
      if (module.HasType<T>())                                                                                         \
        return;                                                                                                        \
                                                                                                                       \
      /* If this type has not been defined in this module, mark it as defined. */                                      \
      module.types_->insert(typeid(T).hash_code());                                                                    \
                                                                                                                       \
      /* Define this type in this module. */                                                                           \
      py::class_<T> cls(module, #TYPE)

//
//
//
#define __ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(TEMPLATE)                                                                    \
                                                                                                                       \
  template <typename... Args>                                                                                          \
  struct __ARIAPython_RecursivelyDefinePythonType<TEMPLATE<Args...>> {                                                 \
    void operator()(const Module &module) {                                                                            \
      using T = TEMPLATE<Args...>;                                                                                     \
                                                                                                                       \
      static_assert(std::is_same_v<T, std::decay_t<T>>,                                                                \
                    "The given type to be defined in Python should be a decayed type");                                \
                                                                                                                       \
      static_assert(!std::is_const_v<T>, "The given type to be defined in Python should not be a const type "          \
                                         "because `const` has no effect in Python");                                   \
      static_assert(!std::is_pointer_v<T>, "The given type to be defined in Python should not be a pointer type "      \
                                           "because pointer types are automatically handled");                         \
                                                                                                                       \
      static_assert(!python::detail::is_python_builtin_type<T>(),                                                      \
                    "The given type to be defined in Python should not be a Python-builtin type");                     \
      static_assert(!python::detail::method<T>, "The given type to be defined in Python should not be a method type"); \
                                                                                                                       \
      /* Return if this type has already been defined in this module. */                                               \
      if (module.HasType<T>())                                                                                         \
        return;                                                                                                        \
                                                                                                                       \
      /* If this type has not been defined in this module, mark it as defined. */                                      \
      module.types_->insert(typeid(T).hash_code());                                                                    \
                                                                                                                       \
      /* Define this type in this module. */                                                                           \
      py::class_<T> cls(module, #TEMPLATE)

//
//
//
// clang-format off
#define __ARIA_PYTHON_TYPE_METHOD_PARAMS2(SPECIFIERS, NAME)                                                            \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  ))                                                                                                                   \
  (T::*)() SPECIFIERS>()(module);                                                                                      \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  ))                                                                                                                   \
  (T::*)() SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS3(SPECIFIERS, NAME, T0)                                                        \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>()))                                                                                                 \
  (T::*)(T0) SPECIFIERS>()(module);                                                                                    \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>()))                                                                                                 \
  (T::*)(T0) SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS4(SPECIFIERS, NAME, T0, T1)                                                    \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (T::*)(T0, T1) SPECIFIERS>()(module);                                                                                \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (T::*)(T0, T1) SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS5(SPECIFIERS, NAME, T0, T1, T2)                                                \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (T::*)(T0, T1, T2) SPECIFIERS>()(module);                                                                            \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (T::*)(T0, T1, T2) SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS6(SPECIFIERS, NAME, T0, T1, T2, T3)                                            \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (T::*)(T0, T1, T2, T3) SPECIFIERS>()(module);                                                                        \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (T::*)(T0, T1, T2, T3) SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS7(SPECIFIERS, NAME, T0, T1, T2, T3, T4)                                        \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>()))                 \
  (T::*)(T0, T1, T2, T3, T4) SPECIFIERS>()(module);                                                                    \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>()))                 \
  (T::*)(T0, T1, T2, T3, T4) SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS8(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5)                                    \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>()))                                                                                                 \
  (T::*)(T0, T1, T2, T3, T4, T5) SPECIFIERS>()(module);                                                                \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>()))                                                                                                 \
  (T::*)(T0, T1, T2, T3, T4, T5) SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS9(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6)                                \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>()))                                                                             \
  (T::*)(T0, T1, T2, T3, T4, T5, T6) SPECIFIERS>()(module);                                                            \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>()))                                                                             \
  (T::*)(T0, T1, T2, T3, T4, T5, T6) SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS10(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6, T7)                           \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>()))                                                         \
  (T::*)(T0, T1, T2, T3, T4, T5, T6, T7) SPECIFIERS>()(module);                                                        \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>()))                                                         \
  (T::*)(T0, T1, T2, T3, T4, T5, T6, T7) SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS11(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8)                       \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>()))                                     \
  (T::*)(T0, T1, T2, T3, T4, T5, T6, T7, T8) SPECIFIERS>()(module);                                                    \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>()))                                     \
  (T::*)(T0, T1, T2, T3, T4, T5, T6, T7, T8) SPECIFIERS>(&T::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS12(SPECIFIERS, NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)                   \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>(), std::declval<T9>()))                 \
  (T::*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) SPECIFIERS>()(module);                                                \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>(), std::declval<T9>()))                 \
  (T::*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) SPECIFIERS>(&T::NAME))
// clang-format on

#define __ARIA_PYTHON_TYPE_METHOD(...)                                                                                 \
  __ARIA_EXPAND(__ARIA_EXPAND(ARIA_CONCAT(__ARIA_PYTHON_TYPE_METHOD_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

//
//
//
#define __ARIA_PYTHON_TYPE_PROPERTY(NAME)                                                                              \
                                                                                                                       \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<T>().NAME()) (T::*)()>()(module);                     \
  __ARIAPython_RecursivelyDefinePythonType<void (T::*)(const decltype(std::declval<T>().ARIA_PROP_IMPL(NAME)()) &)>()( \
      module);                                                                                                         \
                                                                                                                       \
  static_assert(property::detail::PropertyType<decltype(std::declval<T>().NAME())>,                                    \
                "The given type to be defined in Python should be an ARIA property type");                             \
  cls.def_property(                                                                                                    \
      #NAME, static_cast<decltype(std::declval<T>().NAME()) (T::*)()>(&T::NAME),                                       \
      static_cast<void (T::*)(const decltype(std::declval<T>().ARIA_PROP_IMPL(NAME)()) &)>(&T::ARIA_PROP_IMPL(NAME)))

//
//
//
#define __ARIA_PYTHON_TYPE_READONLY_PROPERTY(NAME)                                                                     \
                                                                                                                       \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<T>().NAME()) (T::*)()>()(module);                     \
                                                                                                                       \
  static_assert(property::detail::PropertyType<decltype(std::declval<T>().NAME())>,                                    \
                "The given type to be defined in Python should be an ARIA property type");                             \
  cls.def_property_readonly(#NAME, static_cast<decltype(std::declval<T>().NAME()) (T::*)()>(&T::NAME))

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
  }

//
//
//
#define __ARIA_ADD_PYTHON_TYPE(TYPE, MODULE) ::ARIA::__ARIAPython_RecursivelyDefinePythonType<T>()(MODULE)

//
//
//
//
//
//
//
//
//
// TODO: Move to `PythonSTL.h`.
__ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(std::vector);
__ARIA_PYTHON_TYPE_METHOD(, clear);
__ARIA_PYTHON_TYPE_BINARY_OPERATOR(==);
__ARIA_PYTHON_TYPE_END;

} // namespace ARIA
