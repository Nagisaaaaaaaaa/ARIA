cmake_minimum_required(VERSION 3.25.2)
project(ARIA LANGUAGES CXX)

# User options.
option(ARIA_BUILD_TESTS "Build unit tests" ON)
option(ARIA_USE_64BIT_REAL "using Real = double;" OFF)
option(ARIA_ENABLE_CUDA "Enable CUDA or not (MUST be ON for now)" ON)

option(ARIA_SANITIZE_ADDRESS "Enable GCC/Clang address sanitizer or not" OFF) # To catch out-of-bounds accesses.
option(ARIA_SANITIZE_MEMORY "Enable Clang memory sanitizer or not" OFF) # To catch use of uninitialized memory.
option(ARIA_SANITIZE_THREAD "Enable GCC/Clang thread sanitizer or not" OFF) # To catch multi-threading issues.
option(ARIA_SANITIZE_UNDEFINED "Enable GCC/Clang undefined sanitizer or not" OFF) # To catch undefined-behaviors.

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

#
#
#
# ################################################################################
if ((ARIA_SANITIZE_ADDRESS OR ARIA_SANITIZE_MEMORY OR ARIA_SANITIZE_THREAD OR ARIA_SANITIZE_UNDEFINED) AND
(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)"))
  # Don't optimize too heavily.
  if (U_CMAKE_BUILD_TYPE MATCHES REL)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O1")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -O1")
    add_compile_options(-O1 -fno-optimize-sibling-calls)
  endif ()

  add_compile_options(-fno-omit-frame-pointer)

  if (ARIA_SANITIZE_ADDRESS)
    if (ARIA_ENABLE_CUDA)
      message(AUTHOR_WARNING "When address sanitizer is enabled together with CUDA, one should add \"protect_shadow_gap=0:detect_leaks=0\" to ASAN_OPTIONS")
    endif ()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
    message(STATUS "ARIA: enabling the address sanitizer.")
  endif ()

  if (ARIA_SANITIZE_MEMORY)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      message(FATAL_ERROR "GCC does not support memory sanitizer. Consider disabling ARIA_SANITIZE_MEMORY or use Clang")
    endif ()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=memory")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=memory")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=memory")
    message(STATUS "ARIA: enabling the memory sanitizer.")
  endif ()

  if (ARIA_SANITIZE_THREAD)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
    message(STATUS "ARIA: enabling the thread sanitizer.")
  endif ()

  if (ARIA_SANITIZE_UNDEFINED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=undefined")
    message(STATUS "ARIA: enabling the undefined sanitizer.")
  endif ()
endif ()

add_subdirectory(ARIA)
