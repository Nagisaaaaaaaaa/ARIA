#pragma once

#include "ARIA/Property.h"
#include "ARIA/TypeArray.h"

#include <pybind11/embed.h>
#include <pybind11/operators.h>

#include <list>

namespace ARIA {

namespace python::detail {

namespace py = pybind11;

//
//
//
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
// Whether the given type is a function (global function or static member function) type.
template <typename T>
struct is_function {
  static constexpr bool value = false;
};

template <type_array::detail::NonArrayType Ret, type_array::detail::NonArrayType... Args>
struct is_function<Ret (*)(Args...)> {
  static constexpr bool value = true;

  using return_type = Ret;
  using arguments_types = MakeTypeArray<Args...>;
  using return_and_arguments_types = MakeTypeArray<return_type, arguments_types>;
};

template <typename T>
static constexpr bool is_function_v = is_function<T>::value;

template <typename T>
concept function = is_function_v<T>;

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
  // because these types have been automatically handled.
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
  // void operator()(Module &module) {}
};

//
//
//
namespace python::detail {

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

  /// \brief Define a function for this module.
  ///
  /// \example ```cpp
  /// std::vector<int> add(const std::vector<int> &a, std::vector<int> &b) {
  ///   size_t size = a.size();
  ///   ARIA_ASSERT(size == b.size());
  ///
  ///   std::vector<int> c(size);
  ///   for (size_t i = 0; i < size; ++i)
  ///     c[i] = a[i] + b[i];
  ///
  ///   return c;
  /// }
  ///
  /// module.Def("add0", add)
  ///       .Def("add1", [](const std::vector<int> &a, std::vector<int> &b) { ... });
  /// ```
  template <typename... Ts>
  Module &Def(Ts &&...ts) {
    module_.def(std::forward<Ts>(ts)...);
    return *this;
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

} // namespace python::detail

//
//
//
// Specialization for function types.
template <python::detail::function TFunction>
struct __ARIAPython_RecursivelyDefinePythonType<TFunction> {
  void operator()(python::detail::Module &module) {
    // Define a function will define its return type and all its arguments types.
    ForEach<typename python::detail::is_function<TFunction>::return_and_arguments_types>([&]<typename T>() {
      using TUndecorated = std::remove_const_t<std::remove_pointer_t<std::decay_t<T>>>;

      __ARIAPython_RecursivelyDefinePythonType<TUndecorated>()(module);
    });
  }
};

// Specialization for method types.
template <python::detail::method TMethod>
struct __ARIAPython_RecursivelyDefinePythonType<TMethod> {
  void operator()(python::detail::Module &module) {
    // Define a method will define its return type and all its arguments types.
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
  void operator()(python::detail::Module &module) {
    // Do nothing for Python-builtin types.
  }
};

//
//
//
namespace python::detail {

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

} // namespace python::detail

//
//
//
//
//
//
//
//
//
// `__ARIAPython_RecursivelyDefinePythonType` usually wants to access private methods of a class,
// especially while declaring ARIA properties.
#define __ARIA_PYTHON_TYPE_FRIEND                                                                                      \
  template <typename TUVW>                                                                                             \
  friend struct ::ARIA::__ARIAPython_RecursivelyDefinePythonType;

//
//
//
// Eg: ARIA_PYTHON_TYPE_BEGIN(Object);
#define __ARIA_PYTHON_TYPE_BEGIN(TYPE)                                                                                 \
  /* Specialization for the given type. */                                                                             \
  template <>                                                                                                          \
  struct __ARIAPython_RecursivelyDefinePythonType<TYPE> {                                                              \
    void operator()(python::detail::Module &module) {                                                                  \
      namespace py = python::detail::py;                                                                               \
                                                                                                                       \
      /*! Alias to `T`, in order to make it able to use `T` in other macros. */                                        \
      /*! For example, `ARIA_PYTHON_TYPE_BINARY_OPERATOR(==, decltype(std::declval<T>().value()));`. */                \
      /*! Here, `T` is the type of some complex ARIA property. */                                                      \
      using T = TYPE;                                                                                                  \
                                                                                                                       \
      /* The given type to be defined in Python should be an undecorated type. */                                      \
      static_assert(std::is_same_v<T, std::decay_t<T>>,                                                                \
                    "The given type to be defined in Python should be a decayed type");                                \
      static_assert(!std::is_const_v<T>, "The given type to be defined in Python should not be a const type "          \
                                         "because `const` has no effect in Python");                                   \
      static_assert(!std::is_pointer_v<T>, "The given type to be defined in Python should not be a pointer type "      \
                                           "because pointer types are automatically handled");                         \
                                                                                                                       \
      static_assert(!python::detail::method<T>,                                                                        \
                    "The given type to be defined in Python should not be a method type.");                            \
      static_assert(!python::detail::is_python_builtin_type<T>(),                                                      \
                    "The given type to be defined in Python should not be a Python-builtin type because "              \
                    "these types have been automatically handled");                                                    \
                                                                                                                       \
      /* Return if this type has already been defined in this module. */                                               \
      /*! Non-const references to Python-builtin types have already been checked here. */                              \
      if (module.HasType<T>()) [[likely]]                                                                              \
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
// Eg: ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(std::vector);
#define __ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(TEMPLATE)                                                                    \
  /* Specialization for the given template. */                                                                         \
  template <typename... Args>                                                                                          \
  struct __ARIAPython_RecursivelyDefinePythonType<TEMPLATE<Args...>> {                                                 \
    void operator()(python::detail::Module &module) {                                                                  \
      namespace py = python::detail::py;                                                                               \
                                                                                                                       \
      using T = TEMPLATE<Args...>;                                                                                     \
                                                                                                                       \
      static_assert(std::is_same_v<T, std::decay_t<T>>,                                                                \
                    "The given type to be defined in Python should be a decayed type");                                \
      static_assert(!std::is_const_v<T>, "The given type to be defined in Python should not be a const type "          \
                                         "because `const` has no effect in Python");                                   \
      static_assert(!std::is_pointer_v<T>, "The given type to be defined in Python should not be a pointer type "      \
                                           "because pointer types are automatically handled");                         \
                                                                                                                       \
      static_assert(!python::detail::method<T>,                                                                        \
                    "The given type to be defined in Python should not be a method type.");                            \
      static_assert(!python::detail::is_python_builtin_type<T>(),                                                      \
                    "The given type to be defined in Python should not be a Python-builtin type because "              \
                    "these types have been automatically handled");                                                    \
                                                                                                                       \
      if (module.HasType<T>()) [[likely]]                                                                              \
        return;                                                                                                        \
                                                                                                                       \
      module.types_->insert(typeid(T).hash_code());                                                                    \
                                                                                                                       \
      /* TODO: Give unique name for each instantiated type. */                                                         \
      py::class_<T> cls(module, #TEMPLATE)

//
//
//
// Define a constructor with the given arguments types.
// Eg: ARIA_PYTHON_TYPE_CONSTRUCTOR(int);                     // `T::T(int)`.
//     ARIA_PYTHON_TYPE_CONSTRUCTOR(int, const std::string&); // `T::T(int, const std::string&)`.
//
// In order to support dynamic number of arguments types, "macro magics" are used,
// see the implementation of `ARIA_ASSERT` as a simple example.
//
// For 1 parameter.
//! It is actually impossible for C/C++ macros to accept "zero argument", that is,
//! an empty `__VA_ARGS__` does not mean "zero argument",
//! it actually contains "one argument" (so weird...).
//!
//! In order to support default constructors with "zero argument",
//! this wrapper class is introduced.
//! Eg: `empty_va_args_wrapper_t<>`  is equals to `int`.
//!     `empty_va_args_wrapper_t<T>` is equals to `T`.
template <typename T = int>
struct empty_va_args_wrapper {
  using type = T;
};

template <typename T = int>
using empty_va_args_wrapper_t = typename empty_va_args_wrapper<T>::type;

#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS1(T0)                                                                     \
  /* Define a constructor will define all its arguments types. */                                                      \
  /*! The "zero argument" case is handled here. */                                                                     \
  __ARIAPython_RecursivelyDefinePythonType<                                                                            \
      std::remove_const_t<std::remove_pointer_t<std::decay_t<empty_va_args_wrapper_t<T0>>>>>()(module);                \
  cls.def(py::init<T0>())

// For 2 parameters.
#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS2(T0, T1)                                                                 \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T0>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T1>>>>()(module);    \
  cls.def(py::init<T0, T1>())

// For 3 parameters...
#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS3(T0, T1, T2)                                                             \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T0>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T1>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T2>>>>()(module);    \
  cls.def(py::init<T0, T1, T2>())

#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS4(T0, T1, T2, T3)                                                         \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T0>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T1>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T2>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T3>>>>()(module);    \
  cls.def(py::init<T0, T1, T2, T3>())

#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS5(T0, T1, T2, T3, T4)                                                     \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T0>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T1>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T2>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T3>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T4>>>>()(module);    \
  cls.def(py::init<T0, T1, T2, T3, T4>())

#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS6(T0, T1, T2, T3, T4, T5)                                                 \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T0>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T1>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T2>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T3>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T4>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T5>>>>()(module);    \
  cls.def(py::init<T0, T1, T2, T3, T4, T5>())

#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS7(T0, T1, T2, T3, T4, T5, T6)                                             \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T0>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T1>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T2>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T3>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T4>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T5>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T6>>>>()(module);    \
  cls.def(py::init<T0, T1, T2, T3, T4, T5, T6>())

#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS8(T0, T1, T2, T3, T4, T5, T6, T7)                                         \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T0>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T1>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T2>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T3>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T4>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T5>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T6>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T7>>>>()(module);    \
  cls.def(py::init<T0, T1, T2, T3, T4, T5, T6, T7>())

#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS9(T0, T1, T2, T3, T4, T5, T6, T7, T8)                                     \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T0>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T1>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T2>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T3>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T4>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T5>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T6>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T7>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T8>>>>()(module);    \
  cls.def(py::init<T0, T1, T2, T3, T4, T5, T6, T7, T8>())

#define __ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS10(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)                                \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T0>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T1>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T2>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T3>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T4>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T5>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T6>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T7>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T8>>>>()(module);    \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<T9>>>>()(module);    \
  cls.def(py::init<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>())

#define __ARIA_PYTHON_TYPE_CONSTRUCTOR(...)                                                                            \
  __ARIA_EXPAND(                                                                                                       \
      __ARIA_EXPAND(ARIA_CONCAT(__ARIA_PYTHON_TYPE_CONSTRUCTOR_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

//
//
//
// Define a method with the given specifiers, name, and arguments types.
// Eg: ARIA_PYTHON_TYPE_METHOD(const, value); // `T::value() const`.
//     ARIA_PYTHON_TYPE_METHOD(, value, int); // `T::value(int)`.
//
// clang-format off
// For 2 parameters.
#define __ARIA_PYTHON_TYPE_METHOD_PARAMS2(SPECIFIERS, NAME)                                                            \
  /* Define a method will define its return type and all its arguments types. */                                       \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  ))                                                                                                                   \
  (T::*)() SPECIFIERS>()(module);                                                                                      \
                                                                                                                       \
  /* Calls `py::class_::def` to actually define the method in Python. */                                               \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  ))                                                                                                                   \
  (T::*)() SPECIFIERS>(&T::NAME))

// For 3 parameters.
#define __ARIA_PYTHON_TYPE_METHOD_PARAMS3(SPECIFIERS, NAME, T0)                                                        \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<SPECIFIERS T>().NAME(                                 \
  std::declval<T0>()))                                                                                                 \
  (T::*)(T0) SPECIFIERS>()(module);                                                                                    \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS T>().NAME(                                               \
  std::declval<T0>()))                                                                                                 \
  (T::*)(T0) SPECIFIERS>(&T::NAME))

// For 4 parameters...
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
// Define a static member function with the given name and arguments types.
//
// clang-format off
// For 1 parameter.
#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS1(NAME)                                                               \
  /* Define a function will define its return type and all its arguments types. */                                     \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  ))                                                                                                                   \
  (*)()>()(module);                                                                                                    \
                                                                                                                       \
  /* Calls `cls.def_static` to actually define the function in Python. */                                              \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  ))                                                                                                                   \
  (*)()>(&T::NAME))

// For 2 parameters.
#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS2(NAME, T0)                                                           \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>()))                                                                                                 \
  (*)(T0)>()(module);                                                                                                  \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>()))                                                                                                 \
  (*)(T0)>(&T::NAME))

// For 3 parameters...
#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS3(NAME, T0, T1)                                                       \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (*)(T0, T1)>()(module);                                                                                              \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (*)(T0, T1)>(&T::NAME))

#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS4(NAME, T0, T1, T2)                                                   \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (*)(T0, T1, T2)>()(module);                                                                                          \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (*)(T0, T1, T2)>(&T::NAME))

#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS5(NAME, T0, T1, T2, T3)                                               \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (*)(T0, T1, T2, T3)>()(module);                                                                                      \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (*)(T0, T1, T2, T3)>(&T::NAME))

#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS6(NAME, T0, T1, T2, T3, T4)                                           \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>()))                 \
  (*)(T0, T1, T2, T3, T4)>()(module);                                                                                  \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>()))                 \
  (*)(T0, T1, T2, T3, T4)>(&T::NAME))

#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS7(NAME, T0, T1, T2, T3, T4, T5)                                       \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>()))                                                                                                 \
  (*)(T0, T1, T2, T3, T4, T5)>()(module);                                                                              \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>()))                                                                                                 \
  (*)(T0, T1, T2, T3, T4, T5)>(&T::NAME))

#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS8(NAME, T0, T1, T2, T3, T4, T5, T6)                                   \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>()))                                                                             \
  (*)(T0, T1, T2, T3, T4, T5, T6)>()(module);                                                                          \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>()))                                                                             \
  (*)(T0, T1, T2, T3, T4, T5, T6)>(&T::NAME))

#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS9(NAME, T0, T1, T2, T3, T4, T5, T6, T7)                               \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>()))                                                         \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7)>()(module);                                                                      \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>()))                                                         \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7)>(&T::NAME))

#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS10(NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8)                          \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>()))                                     \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8)>()(module);                                                                  \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>()))                                     \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8)>(&T::NAME))

#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS11(NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)                      \
  __ARIAPython_RecursivelyDefinePythonType<decltype(T::NAME(                                                           \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>(), std::declval<T9>()))                 \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)>()(module);                                                              \
  cls.def_static(#NAME, static_cast<decltype(T::NAME(                                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>(), std::declval<T9>()))                 \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)>(&T::NAME))
// clang-format on

#define __ARIA_PYTHON_TYPE_STATIC_FUNCTION(...)                                                                        \
  __ARIA_EXPAND(                                                                                                       \
      __ARIA_EXPAND(ARIA_CONCAT(__ARIA_PYTHON_TYPE_STATIC_FUNCTION_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

//
//
//
// Define an ARIA property.
// Eg: ARIA_PYTHON_TYPE_PROPERTY(name);
#define __ARIA_PYTHON_TYPE_PROPERTY(NAME)                                                                              \
  /* Define the getter. */                                                                                             \
  /*! Here, `NAME` is used instead of `ARIA_PROP_IMPL(NAME)`, that is, */                                              \
  /*! Python will directly use ARIA property types. */                                                                 \
  /*! This is because the property system of pybind is not that strong, thus */                                        \
  /*! cannot be used as a proxy system. */                                                                             \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<T>().NAME()) (T::*)()>()(module);                     \
                                                                                                                       \
  /* Define the setter. */                                                                                             \
  /*! Here `ARIA_PROP_IMPL(NAME)` is used. */                                                                          \
  __ARIAPython_RecursivelyDefinePythonType<void (T::*)(const decltype(std::declval<T>().ARIA_PROP_IMPL(NAME)()) &)>()( \
      module);                                                                                                         \
                                                                                                                       \
  static_assert(property::detail::PropertyType<decltype(std::declval<T>().NAME())>,                                    \
                "The given type to be defined in Python should be an ARIA property type");                             \
                                                                                                                       \
  /* Calls `py::class_::def_property` to actually define the property in Python. */                                    \
  cls.def_property(                                                                                                    \
      #NAME, static_cast<decltype(std::declval<T>().NAME()) (T::*)()>(&T::NAME),                                       \
      static_cast<void (T::*)(const decltype(std::declval<T>().ARIA_PROP_IMPL(NAME)()) &)>(&T::ARIA_PROP_IMPL(NAME)))

//
//
//
// Similar to `ARIA_PYTHON_TYPE_PROPERTY` but define a readonly ARIA property.
#define __ARIA_PYTHON_TYPE_READONLY_PROPERTY(NAME)                                                                     \
  /* Only need to define the getter. */                                                                                \
  __ARIAPython_RecursivelyDefinePythonType<decltype(std::declval<T>().NAME()) (T::*)()>()(module);                     \
                                                                                                                       \
  static_assert(property::detail::PropertyType<decltype(std::declval<T>().NAME())>,                                    \
                "The given type to be defined in Python should be an ARIA property type");                             \
                                                                                                                       \
  /* Calls `py::class_::def_readonly_property` to actually define the property in Python. */                           \
  cls.def_property_readonly(#NAME, static_cast<decltype(std::declval<T>().NAME()) (T::*)()>(&T::NAME))

//
//
//
// Define a unary operator.
// Eg: ARIA_PYTHON_TYPE_UNARY_OPERATOR(-);
#define __ARIA_PYTHON_TYPE_UNARY_OPERATOR(OPERATOR) cls.def(decltype(OPERATOR py::self)())

//
//
//
// Define a binary operator, which can only operate on the same type.
// Eg: ARIA_PYTHON_TYPE_BINARY_OPERATOR(==);
#define __ARIA_PYTHON_TYPE_BINARY_OPERATOR_PARAMS1(OPERATOR) cls.def(decltype(py::self OPERATOR py::self)())

// Define a binary operator, which can operate on other types.
// Eg: ARIA_PYTHON_TYPE_BINARY_OPERATOR(+, int);
#define __ARIA_PYTHON_TYPE_BINARY_OPERATOR_PARAMS2(OPERATOR, OTHERS)                                                   \
  static_assert(!std::is_same_v<std::remove_const_t<std::remove_pointer_t<std::decay_t<T>>>,                           \
                                std::remove_const_t<std::remove_pointer_t<std::decay_t<OTHERS>>>>,                     \
                "Omit the second argument and use something like `ARIA_PYTHON_TYPE_BINARY_OPERATOR(==)` instead, if "  \
                "you only want to operate on the same type.");                                                         \
                                                                                                                       \
  /* Recursively define `OTHERS` in Python. */                                                                         \
  __ARIAPython_RecursivelyDefinePythonType<std::remove_const_t<std::remove_pointer_t<std::decay_t<OTHERS>>>>()(        \
      module);                                                                                                         \
                                                                                                                       \
  /* Define all the 3 variants. */                                                                                     \
  cls.def(decltype(py::self OPERATOR py::self)());                                                                     \
  cls.def(decltype(py::self OPERATOR std::declval<OTHERS>())());                                                       \
  cls.def(decltype(std::declval<OTHERS>() OPERATOR py::self)())

#define __ARIA_PYTHON_TYPE_BINARY_OPERATOR(...)                                                                        \
  __ARIA_EXPAND(                                                                                                       \
      __ARIA_EXPAND(ARIA_CONCAT(__ARIA_PYTHON_TYPE_BINARY_OPERATOR_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

//
//
//
// Manually add Python functions for a given module.
// Eg: ARIA_PYTHON_ADD_FUNCTION(module, add, int, int);
//
// clang-format off
// For 2 parameter.
#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS2(MODULE, NAME)                                                               \
  /* Define a function will define its return type and all its arguments types. */                                     \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  ))                                                                                                                   \
  (*)()>()(MODULE);                                                                                                    \
                                                                                                                       \
  /* Calls `module.Def` to actually define the function in Python. */                                                  \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  ))                                                                                                                   \
  (*)()>(&NAME))

// For 3 parameters.
#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS3(MODULE, NAME, T0)                                                           \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>()))                                                                                                 \
  (*)(T0)>()(MODULE);                                                                                                  \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>()))                                                                                                 \
  (*)(T0)>(&NAME))

// For 4 parameters...
#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS4(MODULE, NAME, T0, T1)                                                       \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (*)(T0, T1)>()(MODULE);                                                                                              \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (*)(T0, T1)>(&NAME))

#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS5(MODULE, NAME, T0, T1, T2)                                                   \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (*)(T0, T1, T2)>()(MODULE);                                                                                          \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (*)(T0, T1, T2)>(&NAME))

#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS6(MODULE, NAME, T0, T1, T2, T3)                                               \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (*)(T0, T1, T2, T3)>()(MODULE);                                                                                      \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (*)(T0, T1, T2, T3)>(&NAME))

#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS7(MODULE, NAME, T0, T1, T2, T3, T4)                                           \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>()))                 \
  (*)(T0, T1, T2, T3, T4)>()(MODULE);                                                                                  \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>()))                 \
  (*)(T0, T1, T2, T3, T4)>(&NAME))

#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS8(MODULE, NAME, T0, T1, T2, T3, T4, T5)                                       \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>()))                                                                                                 \
  (*)(T0, T1, T2, T3, T4, T5)>()(MODULE);                                                                              \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>()))                                                                                                 \
  (*)(T0, T1, T2, T3, T4, T5)>(&NAME))

#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS9(MODULE, NAME, T0, T1, T2, T3, T4, T5, T6)                                   \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>()))                                                                             \
  (*)(T0, T1, T2, T3, T4, T5, T6)>()(MODULE);                                                                          \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>()))                                                                             \
  (*)(T0, T1, T2, T3, T4, T5, T6)>(&NAME))

#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS10(MODULE, NAME, T0, T1, T2, T3, T4, T5, T6, T7)                              \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>()))                                                         \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7)>()(MODULE);                                                                      \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>()))                                                         \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7)>(&NAME))

#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS11(MODULE, NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8)                          \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>()))                                     \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8)>()(MODULE);                                                                  \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>()))                                     \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8)>(&NAME))

#define __ARIA_PYTHON_ADD_FUNCTION_PARAMS12(MODULE, NAME, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)                      \
  __ARIAPython_RecursivelyDefinePythonType<decltype(NAME(                                                              \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>(), std::declval<T9>()))                 \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)>()(MODULE);                                                              \
  MODULE.Def(#NAME, static_cast<decltype(NAME(                                                                         \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>(), std::declval<T4>(),                  \
  std::declval<T5>(), std::declval<T6>(), std::declval<T7>(), std::declval<T8>(), std::declval<T9>()))                 \
  (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)>(&NAME))
// clang-format on

#define __ARIA_PYTHON_ADD_FUNCTION(...)                                                                                \
  __ARIA_EXPAND(__ARIA_EXPAND(ARIA_CONCAT(__ARIA_PYTHON_ADD_FUNCTION_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

// Manually add Python functions as dependency for a given class.
// Eg: ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(add, int, int);
#define __ARIA_PYTHON_TYPE_EXTERNAL_FUNCTION(...) __ARIA_EXPAND(__ARIA_PYTHON_ADD_FUNCTION(module, __VA_ARGS__))

//
//
//
#define __ARIA_PYTHON_TYPE_END                                                                                         \
  }                                                                                                                    \
  }

//
//
//
// Manually add Python types for `module`.
// This version is helpful when used within `ARIA_PYTHON_TYPE_BEGIN` and `ARIA_PYTHON_TYPE_END`.
// Eg: ARIA_PYTHON_ADD_TYPE_PARAMS1(Object);
#define __ARIA_PYTHON_ADD_TYPE_PARAMS1(TYPE) ::ARIA::__ARIAPython_RecursivelyDefinePythonType<TYPE>()(module)

// Manually add Python types for a given module.
// Eg: ARIA_PYTHON_ADD_TYPE_PARAMS1(Object, module);
#define __ARIA_PYTHON_ADD_TYPE_PARAMS2(TYPE, MODULE) ::ARIA::__ARIAPython_RecursivelyDefinePythonType<TYPE>()(MODULE)

#define __ARIA_PYTHON_ADD_TYPE(...)                                                                                    \
  __ARIA_EXPAND(__ARIA_EXPAND(ARIA_CONCAT(__ARIA_PYTHON_ADD_TYPE_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

//
//
//
//
//
//
//
//
//
// TODO: Fully define commonly used STL types.

// std::vector
__ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(std::vector);
__ARIA_PYTHON_TYPE_METHOD(, clear);
__ARIA_PYTHON_TYPE_METHOD(const, size);
__ARIA_PYTHON_TYPE_BINARY_OPERATOR(==);
__ARIA_PYTHON_TYPE_END;

} // namespace ARIA
