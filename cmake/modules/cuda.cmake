cmake_minimum_required(VERSION 3.25.2)

# This script setup CUDA.
message(STATUS "ARIA:  Configuring CUDA...")

# ################################################################################
# Basic CMake setup.
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if (NOT CUDAToolkit_FOUND)
  message(FATAL_ERROR "CUDAToolkit is mandatory for ARIA, please install CUDA first.")
endif ()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set NVCC executable for non-CMake targets.
set(CUDA_NVCC_EXECUTABLE "${CUDAToolkit_BIN_DIR}/nvcc")

# Prepare CUDA libraries linkage.
list(APPEND ARIA_CUDA_COMMON_LIBS "CUDA::cudart")

# ################################################################################
# Advanced library setup.
# Add CUDA related libraries here

# ################################################################################
# Fix compilation issues.
if (CMAKE_CUDA_COMPILER MATCHES "(NVIDIA|nvcc)")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

  # Compatibility for older GCC host compiler and older CMake.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++20")

  # Warning #20208-D: 'long double' is treated as 'double' in device code.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20208")

  # Embed lineinfo for nsight-compute only in Debug mode.
  if (CMAKE_BUILD_TYPE MATCHES "Debug")
    message(AUTHOR_WARNING "Debug info is embedded for CUDA profiler, consider modify cmake/modules/cuda.cmake to disable it")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
  endif ()
endif ()

# Enable separable compilation for device function linkage.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ################################################################################
# Set CUDA architecture.
include(select_compute_arch)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARIA_CUDA_ARCH)
set(CMAKE_CUDA_ARCHITECTURES "${ARIA_CUDA_ARCH_archs}" CACHE STRING "CUDA architectures" FORCE)

# ################################################################################
# Inherit CUDA flags from CXX flags (for example, the -march=native).
if (WIN32)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\" ${CMAKE_CXX_FLAGS}\"")
else ()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CXX_FLAGS}")
endif ()

# ################################################################################
# Log the setup result.
message(STATUS "ARIA:    CUDA is configured with")
message(STATUS "         CUDA_NVCC_EXECUTABLE: ${CUDA_NVCC_EXECUTABLE}")
message(STATUS "         ARIA_CUDA_COMMON_LIBS: ${ARIA_CUDA_COMMON_LIBS}")
