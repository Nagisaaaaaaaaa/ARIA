#pragma once

/// \file
/// \brief This file introduces a coroutine scheduler whose workflow is very like a "bus station".
/// That is, a `station` is where passengers (coroutines) suspend and
/// wait until the bus driver (a manager thread) comes and goes (resume).
///
/// `station` makes it easier to build an event system, for example,
/// you may want to do something when some mouse button is pressed.
/// It may be a good choice to use `station` for the abstraction of the mouse button events.

//
//
//
//
//
#include "ARIA/Coroutine/detail/SchedulerBase.h"

#include <coroutine>
#include <list>
#include <mutex>

namespace ARIA::Coroutine {

/// \brief A coroutine scheduler whose workflow is very like a "bus station".
/// That is, a `station` is where passengers (coroutines) suspend and
/// wait until the bus driver (a manager thread) comes and goes (resume).
///
/// \warning `station` is implemented with policy-based design.
/// Whether the member functions are thread-safe and the overall performance are
/// based on the template parameter `TMutex`.
///
/// By default, `TMutex` is set to `BrokenMutex`, means all the member functions are non thread-safe.
/// When `TMutex` is set to a "good" mutex, all member functions are thread-safe.
/// That is, passengers may come from many threads and concurrently wait the bus.
///
/// \warning After go, all passengers resume on the bus driver thread, not their original threads.
template <typename TMutex = BrokenMutex>
class station : public SchedulerBase<TMutex> {
public:
  /// \brief Called by the bus driver (a manager thread).
  /// Take all the suspended (waiting) coroutines from the bus station (the waiting list) and go (resume).
  ///
  /// \example ```cpp
  /// station.go();
  /// ```
  ///
  /// \warning Passengers (coroutines) will be resumed on the bus driver thread, not their original threads.
  ///
  /// You can even change the bus driver (change the manager thread),
  /// thread-safety is also controlled by `TMutex`.
  /// Note that passengers will be resumed on the new bus driver thread.
  /// So, pay attention if you want to do this.
  inline void go();

  /// \brief Called by the passengers (the coroutines who want to suspend and wait the bus come).
  /// The passengers call `co_await station.schedule()` to suspend, and
  /// wait until the bus driver (a manager thread) calls `station.go()`.
  /// Then, the bus goes (resume) with all the passengers.
  ///
  /// \example ```cpp
  /// co_await station.schedule();
  /// ```
  ///
  /// \warning When the `station` destruct, all the passengers waiting the bus will be killed (destroyed).
  /// Their coroutine contexts will be destroyed and
  /// all the corresponding destructors of all the local variables will be called.
  inline auto schedule() noexcept;

  //
  //
  //
private:
  using Base = SchedulerBase<TMutex>;

public:
  using Base::Base;

  ARIA_COPY_MOVE_ABILITY(station, delete, delete);

  /// \warning When the `station` destruct, all the passengers waiting the bus will be killed (destroyed).
  /// Their coroutine contexts will be destroyed and
  /// all the corresponding destructors of all the local variables will be called.
  inline ~station() noexcept;

  //
  //
  //
  //
  //
private:
  TMutex mutex_{};
  static_assert(noexcept(mutex_.lock()) && noexcept(mutex_.unlock()),
                "Methods `lock` and `unlock` of the station mutex have to be `noexcept`, "
                "to make it able to implement a `noexcept` destructor.");
  //! See the implementation of the destructor to understand:
  //!   1. Why we MUST acquire the lock and ensure thread-safety in the destructor,
  //!   2. Why simply writing `std::lock_guard guard{mutex_};` is not enough, and
  //!      why we have to inherit from `SchedulerBase<TMutex>`.

  std::list<std::coroutine_handle<>> passengers_{};

  //
  //
  //
private:
  using Base::validity;
};

} // namespace ARIA::Coroutine

//
//
//
//
//
#include "ARIA/Coroutine/detail/Station.inc"
