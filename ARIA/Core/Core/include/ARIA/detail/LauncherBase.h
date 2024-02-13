#pragma once

#include "ARIA/Layout.h"

#include <cuda/api.hpp>

namespace ARIA {

namespace launcher::detail {

//! As introduced in `Launcher.h`, `Launcher` is implemented with the builder pattern.
//! There are builder methods such as `overallSize()` and `blockSize()`.
//! Actually, there methods are implemented with cuda-api-wrappers,
//! see https://github.com/eyalroz/cuda-api-wrappers.
//! ARIA just fetches these methods with inheritance and renaming.
//! Methods fetching is implemented in `LaunchBase`, which is a CRTP-based class.
//! All specialization of `Launcher` should inherit from `LaunchBase` to
//! inherit the builder methods.
//!
//! However, class `Launcher` should not inherit all builder methods.
//! For example, the `Launcher` constructed with a `Layout` should not inherit `overallSize()`,
//! since the overall size is implicitly contained in the layout.
//! That is why builder methods are fetched as `protected`, not `public`.
//! That is also why the derived class `Launcher` should
//! explicit `using` all the builder methods it want and discard the remaining.

//
//
//
//
//
// The kernel for launching integrals.
template <std::integral TIdx, typename F>
ARIA_KERNEL static void KernelLaunchIdx(TIdx size, F f) {
  TIdx i = static_cast<TIdx>(threadIdx.x) + static_cast<TIdx>(blockIdx.x) * static_cast<TIdx>(blockDim.x);
  if (i >= size)
    return;

  f(i);
}

// The kernel for launching layouts.
template <layout::detail::LayoutType TLayout, typename F>
ARIA_KERNEL static void KernelLaunchLayout(TLayout layout, int cosize_safe, F f) {
  int i = static_cast<int>(threadIdx.x) + static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
  if (i >= cosize_safe)
    return;

  // Whether `f` can be called with `f(Coord{x, y, ...})`.
  if constexpr (std::is_invocable_v<F, decltype(layout.get_hier_coord(i))>)
    f(layout.get_hier_coord(i));
  // Whether `f` can be called with `f(i)`.
  else if constexpr (std::is_invocable_v<F, int>)
    f(i);
  // Whether `f` can be called with `f(x, y, ...)`.
  else
    cute::apply(layout.get_hier_coord(i), std::forward<F>(f));
}

//
//
//
//
//
// This macro wraps the builder method from `cuda::launch_config_builder_t`:
//   1. Rename the method,
//   2. Return `Launcher&` instead of `cuda::launch_config_builder_t&`.
#define __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(newName, oldName)                                                  \
  template <typename... Ts>                                                                                            \
  decltype(auto) newName(Ts &&...ts) {                                                                                 \
    Base::oldName(std::forward<Ts>(ts)...);                                                                            \
    return derived();                                                                                                  \
  }

//
//
//
// `LauncherBase` inherits from `cuda::launch_config_builder_t` to fetch its methods.
template <typename TDerived>
class LauncherBase : private cuda::launch_config_builder_t {
private:
  ARIA_HOST_DEVICE TDerived &derived() { return *static_cast<TDerived *>(this); }

  ARIA_HOST_DEVICE const TDerived &derived() const { return *static_cast<const TDerived *>(this); }

private:
  using Base = cuda::launch_config_builder_t;

protected:
  // Block size is set to 256 by default.
  static constexpr uint blockSizeDefault = 256;

  // Setup defaults.
  LauncherBase() { blockSize(blockSizeDefault); }

  ARIA_COPY_MOVE_ABILITY(LauncherBase, default, default);

protected:
  // Fetch methods.
  // clang-format off
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(dimensions, dimensions)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(blockDimensions, block_dimensions)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(blockSize, block_size)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(useMaximumLinearBlock, use_maximum_linear_block)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(gridDimensions, grid_dimensions)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(gridSize, grid_size)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(numBlocks, num_blocks)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(overallDimensions, overall_dimensions)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(overallSize, overall_size)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(blockCooperation, block_cooperation)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(blocksMayCooperate, blocks_may_cooperate)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(blocksDontCooperate, blocks_dont_cooperate)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(dynamicSharedMemorySize, dynamic_shared_memory_size)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(noDynamicSharedMemory, no_dynamic_shared_memory)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(dynamicSharedMemory, dynamic_shared_memory)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(kernel, kernel)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(kernelIndependent, kernel_independent)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(noKernel, no_kernel)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(saturateWithActiveBlocks, saturate_with_active_blocks)
  __ARIA_LAUNCHER_BASE_WRAP_BUILDER_INTERFACE(minParamsForMaxOccupancy, min_params_for_max_occupancy)
  // clang-format on

protected:
  // A wrapper function for `cuda::launch`.
  template <typename Kernel, typename... Args>
  void Launch(Kernel &&kernel, Args &&...args) {
    cuda::launch(std::forward<Kernel>(kernel), Base::build(), std::forward<Args>(args)...);
  }
};

} // namespace launcher::detail

} // namespace ARIA
