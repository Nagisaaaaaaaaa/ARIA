#pragma once

#include "ARIA/Property.h"
#include "ARIA/detail/BrokenMutex.h"

#include <atomic>

namespace ARIA::Coroutine {

//! A main difference between coroutines and threads is that
//! it is easy to terminate (destroy) a coroutine, but rather difficult to terminate a thread.
//! ```cpp
//! std::coroutine_handle<> coroutine;
//! coroutine.destroy(); // This function destroys a coroutine.
//! ```
//!
//! So, it is common that schedulers are destroyed earlier than the coroutines inside them.
//! In the first sight, there are two design strategies:
//!   1. Resume all the coroutines currently inside the scheduler,
//!   2. Destroy all the coroutines currently inside the scheduler.
//!
//! However, the first one will not work, since all the coroutines
//! will be resumed in the destructors of the schedulers.
//! We know that destructors should be designed as `noexcept`, but
//! `resume` is not guaranteed to be `noexcept`.
//!
//! Since `destroy` is `noexcept`, only the second one may work.
//! That is, we must destroy all the coroutines.
//! Then, thread-safety matters, because while the destructor of the scheduler is executing,
//! other threads may be simultaneously calling `co_await` on this scheduler.
//! This makes things extremely complex because even a single mutex is not enough.
//! See the comments of `station` for the detailed reasons.
//! We have to introduce an atomic `validity`, which indicates
//! whether the scheduler is still valid or not.
//!
//! Another problem is that schedulers are implemented with policy-based design.
//! Whether a scheduler is thread-safe based on the template parameter `TMutex`.
//! For the default parameter `BrokenMutex`, it is not necessary for `validity` to be atomic.
//! So, template specialization is used in the following codes.
//! This is also the mean reason why we need a base class.

//
//
//
//
//
/// \brief The base class of all the ARIA schedulers.
/// This class is necessary in order to make destructors of the schedulers safe enough.
template <typename TMutex>
class SchedulerBase {
protected:
  /// \brief Whether the scheduler is still valid or not.
  /// \note Getter and setter of this property is atomic and `noexcept`.
  ARIA_PROP(public, public, , bool, validity);

  //
public:
  SchedulerBase() = default;
  ARIA_COPY_MOVE_ABILITY(SchedulerBase, delete, delete);

  //
private:
  std::atomic<bool> validity_{true};

  //
private:
  [[nodiscard]] bool ARIA_PROP_IMPL(validity)() const noexcept { return validity_.load(std::memory_order_acquire); }

  void ARIA_PROP_IMPL(validity)(const bool &value) noexcept { validity_.store(value, std::memory_order_release); }
};

//
//
//
/// \brief The specialization for `BrokenMutex`, where `validity` is non-atomic.
template <>
class SchedulerBase<BrokenMutex> {
protected:
  /// \brief Whether the scheduler is still valid or not.
  /// \note Getter and setter of this property is non-atomic but `noexcept`.
  ARIA_REF_PROP(public, , validity, validity_);

  //
public:
  SchedulerBase() = default;
  ARIA_COPY_MOVE_ABILITY(SchedulerBase, delete, delete);

  //
private:
  bool validity_{true};
};

}; // namespace ARIA::Coroutine
