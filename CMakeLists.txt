cmake_minimum_required(VERSION 3.25.2)
project(ARIA LANGUAGES CXX)

# User options.
option(ARIA_BUILD_TESTS "Build unit tests" ON)
option(ARIA_USE_64BIT_REAL "using Real = double;" OFF)

list(APPEND CMAKE_MODULE_PATH "${ARIA_SOURCE_DIR}/cmake/modules" "${CMAKE_BINARY_DIR}")

#
#
#
# ################################################################################

# Setup flags for Windows.
if ((WIN32) AND (MSVC) AND (MSVC_VERSION GREATER_EQUAL 1914))
  string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus /MP /utf-8")
endif ()

# In-tree builds are not permitted.
include(ARIA_ForceOutOfTreeBuild)
aria_force_out_of_tree_build()

# Somehow when using GCC and Clang on Linux, it requires fmt and spdlog to be compiled
# with -fPIC. Solely setting up CMAKE_POSITION_INDEPENDENT_CODE is not enough.
if (CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fPIC>")
endif ()

# add_test() supports arbitrary characters.
if (POLICY CMP0110)
  cmake_policy(SET CMP0110 NEW)
endif ()

if (ARIA_USE_64BIT_REAL)
  set(aria_real_type double)
  add_definitions(-DARIA_USE_64BIT_REAL)
else ()
  set(aria_real_type float)
  add_definitions(-DARIA_USE_32BIT_REAL)
endif ()

#
#
#
# ################################################################################
include(CPM)

CPMAddPackage(
    GITHUB_REPOSITORY google/googletest
    VERSION 1.14.0
    OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt" "BUILD_GMOCK OFF"
)

#
#
#
# ################################################################################
include(cuda)

#
#
#
# ################################################################################
if (ARIA_BUILD_TESTS)
  enable_testing()
endif ()

add_subdirectory(ARIA)
