configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ARIA/ARIA.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ARIA/ARIA.h"
)

# Setup the module with module name.
aria_setup_module(Core Core)

CPMAddPackage(
    NAME boost-assert
    GITHUB_REPOSITORY boostorg/assert
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-bind
    GITHUB_REPOSITORY boostorg/bind
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-concept-check
    GITHUB_REPOSITORY boostorg/concept_check
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-config
    GITHUB_REPOSITORY boostorg/config
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-container-hash
    GITHUB_REPOSITORY boostorg/container_hash
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-core
    GITHUB_REPOSITORY boostorg/core
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-describe
    GITHUB_REPOSITORY boostorg/describe
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-detail
    GITHUB_REPOSITORY boostorg/detail
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-function
    GITHUB_REPOSITORY boostorg/function
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-function-types
    GITHUB_REPOSITORY boostorg/function_types
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-functional
    GITHUB_REPOSITORY boostorg/functional
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-fusion
    GITHUB_REPOSITORY boostorg/fusion
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-io
    GITHUB_REPOSITORY boostorg/io
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-iterator
    GITHUB_REPOSITORY boostorg/iterator
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-move
    GITHUB_REPOSITORY boostorg/move
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-mp11
    GITHUB_REPOSITORY boostorg/mp11
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-mpl
    GITHUB_REPOSITORY boostorg/mpl
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-optional
    GITHUB_REPOSITORY boostorg/optional
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-pfr
    GITHUB_REPOSITORY boostorg/pfr
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-predef
    GITHUB_REPOSITORY boostorg/predef
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-preprocessor
    GITHUB_REPOSITORY boostorg/preprocessor
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-smart-ptr
    GITHUB_REPOSITORY boostorg/smart_ptr
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-static-assert
    GITHUB_REPOSITORY boostorg/static_assert
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-throw-exception
    GITHUB_REPOSITORY boostorg/throw_exception
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-tuple
    GITHUB_REPOSITORY boostorg/tuple
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-type-traits
    GITHUB_REPOSITORY boostorg/type_traits
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-typeof
    GITHUB_REPOSITORY boostorg/typeof
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME boost-utility
    GITHUB_REPOSITORY boostorg/utility
    GIT_TAG boost-1.87.0
)

CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG "10.1.1"
)

#CPMAddPackage(
#    NAME LuaJIT
#    GITHUB_REPOSITORY WohlSoft/LuaJIT
#    GIT_TAG v2.1
#)

CPMAddPackage(
    NAME pybind11
    GITHUB_REPOSITORY pybind/pybind11
    GIT_TAG v2.11.1
)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
target_include_directories(pybind11::pybind11 INTERFACE ${Python_INCLUDE_DIRS})
target_link_libraries(pybind11::pybind11 INTERFACE ${Python_LIBRARIES})

CPMAddPackage(
    NAME small_vector
    GITHUB_REPOSITORY gharveymn/small_vector
    GIT_TAG main
)

#CPMAddPackage(
#    NAME sol2
#    GITHUB_REPOSITORY ThePhD/sol2
#    GIT_TAG v3.3.1
#)

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    VERSION "1.12.0"
    OPTIONS "SPDLOG_FMT_EXTERNAL_HO ON"
)

CPMAddPackage(
    NAME cuda-api-wrappers
    GIT_TAG v0.6.7
    GITHUB_REPOSITORY eyalroz/cuda-api-wrappers
    OPTIONS "CAW_BUILD_EXAMPLES On"
)

CPMAddPackage(
    NAME cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v3.4.0

    DOWNLOAD_ONLY YES
)
add_library(CUTLASS INTERFACE IMPORTED)
target_include_directories(CUTLASS INTERFACE ${cutlass_SOURCE_DIR}/include)

# Add the module headers.
set(core_core_headers
    ARIA/ARIA.h
    ARIA/Auto.h
    ARIA/BFloat16.h
    ARIA/BitArray.h
    ARIA/BitVector.h
    ARIA/Buyout.h
    ARIA/Constant.h
    ARIA/DisjointSet.h
    ARIA/Float16.h
    ARIA/ForEach.h
    ARIA/Invocations.h
    ARIA/Launcher.h
    ARIA/Layout.h
    ARIA/Let.h
    ARIA/Mosaic.h
    ARIA/Property.h
    ARIA/PropertySTL.h
    ARIA/Python.h
    ARIA/Registry.h
    ARIA/SmallVector.h
    ARIA/TensorVector.h
    ARIA/Tup.h
    ARIA/TypeArray.h
    ARIA/TypeSet.h
    ARIA/Vector.h
)

# Add the module sources (if any).
set(core_core_sources
)

# Add the module.
aria_add_module(
    Core Core
    HEADERS ${core_core_headers}
    SOURCES ${core_core_sources}
    HARD_DEPENDENCIES
    Boost::iterator
    Boost::pfr
    CUDA::cudart
    fmt::fmt
    #libluajit
    pybind11::pybind11
    small_vector
    #sol2::sol2
    spdlog::spdlog
    cuda-api-wrappers::runtime-and-driver
    CUTLASS
    CMAKE_SUBDIRS tests
)

# Set the module properties.
#target_compile_definitions(${ARIA_MODULE_NAME} INTERFACE ARIA_BUILD_CORE_CORE)
target_compile_features(${ARIA_MODULE_NAME}
    INTERFACE
    cxx_constexpr
    cxx_noexcept
    cxx_variable_templates
    cxx_std_20
)

# Enable most compilation check and -Werror on GCC and clang.
#if (CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
#  target_compile_options(${ARIA_MODULE_NAME} INTERFACE -Wall -Wextra -Wpedantic -Werror)
#endif ()
