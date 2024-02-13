#pragma once

/// \brief A `Launcher` abstracts the configurations and launching of a CUDA kernel.
///
/// `Launcher` makes it easier to launch a kernel, functor, or lambda function.
/// Users no longer need to write the tedious `<<<...>>>`,
/// but can also get the finest control of the launching process.

//
//
//
//
//
#include "ARIA/detail/LauncherBase.h"

namespace ARIA {

/// \brief A `Launcher` abstracts the configurations and launching of a CUDA kernel.
///
/// `Launcher` is implement in builder pattern.
/// Follow these steps to launch a kernel:
///   1. Select which kernel, functor, or lambda function to launch,
///   2. (Maybe optionally) configure the launch parameters,
///   3. Calls `.Launch(...)` to launch the kernel.
///
/// `Launcher` is implemented with template specialization.
/// There are many ways to launch a kernel, see the following examples.
template <typename... Ts>
class Launcher;

//
//
//
/// \brief Explicitly launch a `__global__` (ARIA_KERNEL) function.
///
/// \example ```cpp
/// ARIA_KERNEL static void KernelInit(int size, int *data) {
///   int i = static_cast<int>(threadIdx.x) + static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
///   if (i >= size)
///     return;
///
///   data[i] = i;
/// }
///
/// int size = 10;
/// thrust::device_vector<int> v(size);
///
/// // `overallSize` is required, `blockSize` is optional.
/// Launcher(KernelInit).overallSize(size).blockSize(256).Launch(size, v.data().get());
///
/// cuda::device::current::get().synchronize();
///
/// for (int i = 0; i < size; ++i)
///   EXPECT_TRUE(v[i] == i);
/// ```
template <typename Kernel>
class Launcher<Kernel> : public launcher::detail::LauncherBase<Launcher<Kernel>> {
private:
  using Base = launcher::detail::LauncherBase<Launcher<Kernel>>;

public:
  explicit Launcher(const Kernel &kernel) : kernel_(kernel) {}

  ARIA_COPY_MOVE_ABILITY(Launcher, default, default);

public:
  using Base::blockSize;
  using Base::overallSize;

  template <typename... Args>
  void Launch(Args &&...args) {
    Base::Launch(kernel_, std::forward<Args>(args)...);
  }

private:
  const Kernel &kernel_;
};

template <typename Kernel>
Launcher(const Kernel &kernel) -> Launcher<Kernel>;

//
//
//
/// \brief Launch a functor or a lambda function with an integral.
///
/// \example ```cpp
/// int size = 10;
/// thrust::device_vector<int> v(size);
///
/// // `blockSize` is optional.
/// Launcher(size, [v = v.data().get()] ARIA_DEVICE(int i) { v[i] = i; }).blockSize(256).Launch();
///
/// cuda::device::current::get().synchronize();
///
/// for (int i = 0; i < size; ++i)
///   EXPECT_TRUE(v[i] == i);
/// ```
template <std::integral TIdx, typename F>
class Launcher<TIdx, F> : public launcher::detail::LauncherBase<Launcher<TIdx, F>> {
private:
  using Base = launcher::detail::LauncherBase<Launcher<TIdx, F>>;

public:
  Launcher(const TIdx &size, const F &f) : size_(size), f_(f) { Base::overallSize(size_); }

  ARIA_COPY_MOVE_ABILITY(Launcher, default, default);

public:
  using Base::blockSize;

  void Launch() { Base::Launch(launcher::detail::KernelLaunchIdx<TIdx, F>, size_, f_); }

private:
  TIdx size_;
  F f_;
};

template <std::integral TIdx, typename F>
Launcher(const TIdx &size, const F &f) -> Launcher<TIdx, F>;

//
//
//
/// \brief Launch a functor or a lambda function with a `Layout`.
///
/// \example ```cpp
/// auto layout = make_layout_major(5, 6);
///
/// auto aD = make_tensor_vector<int, SpaceDevice>(layout);
/// auto bD = make_tensor_vector<int, SpaceDevice>(layout);
///
/// for (int i = 0; i < aD.size<0>(); ++i) {
///   for (int k = 0; k < aD.size<1>(); ++k) {
///     aD(i, k) = 2 * i + 2 * k;
///     bD(i, k) = i - k;
///   }
/// }
///
/// auto cD = make_tensor_vector<int, SpaceDevice>(layout);
///
/// Launcher(aD.layout(), [a = aD.tensor(), b = bD.tensor(), c = cD.tensor()]
///                       ARIA_DEVICE(const int &x, const int &y) {
///   c(x, y) = a(x, y) + b(x, y);})
/// .blockSize(128)
/// .Launch();
///
/// cuda::device::current::get().synchronize();
///
/// for (int i = 0; i < cD.size<0>(); ++i) {
///   for (int k = 0; k < cD.size<1>(); ++k) {
///     EXPECT_TRUE(cD(i, k) == 3 * i + k);
///   }
/// }
/// ```
template <layout::detail::LayoutType TLayout, typename F>
class Launcher<TLayout, F> : public launcher::detail::LauncherBase<Launcher<TLayout, F>> {
private:
  using Base = launcher::detail::LauncherBase<Launcher<TLayout, F>>;

public:
  Launcher(const TLayout &layout, const F &f) : layout_(layout), f_(f) { Base::overallSize(cosize_safe(layout_)); }

  ARIA_COPY_MOVE_ABILITY(Launcher, default, default);

public:
  using Base::blockSize;

  void Launch() { Base::Launch(launcher::detail::KernelLaunchLayout<TLayout, F>, layout_, cosize_safe(layout_), f_); }

private:
  TLayout layout_;
  F f_;
};

template <layout::detail::LayoutType TLayout, typename F>
Launcher(const TLayout &layout, const F &f) -> Launcher<TLayout, F>;

} // namespace ARIA
