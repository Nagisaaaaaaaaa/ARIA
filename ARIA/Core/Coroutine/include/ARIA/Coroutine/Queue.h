#pragma once

/// \file
/// \brief This file introduces a coroutine scheduler whose workflow is very like a "queue".
/// That is, a `queue` is where coroutines suspend and wait until
/// a manager thread is ready to resume them.
///
/// `queue` makes it easier to build an event system, for example,
/// if you want to implement a game similar to "shadowverse", see https://shadowverse.com/,
/// you may need such an event system to recursively handle the card effects.

//
//
//
//
//
#include "ARIA/Coroutine/detail/SchedulerBase.h"

#include <coroutine>
#include <mutex>
#include <queue>

namespace ARIA::Coroutine {

/// \brief A coroutine scheduler whose workflow is very like a "queue".
/// That is, a `queue` is where coroutines suspend and wait until
/// a manager thread is ready to resume them.
///
/// \warning `queue` is implemented with policy-based design.
/// Whether the member functions are thread-safe and the overall performance are
/// based on the template parameter `TMutex`.
///
/// By default, `TMutex` is set to `BrokenMutex`, means all the member functions are non thread-safe.
/// When `TMutex` is set to a "good" mutex, all member functions are thread-safe.
/// That is, coroutines may come from many threads and concurrently wait the queue.
///
/// \warning After a coroutine is popped from the queue, it resumes on the queue thread, not its original thread.
template <typename TMutex = BrokenMutex>
class queue : public SchedulerBase<TMutex> {
public:
  /// \brief Called by a manager thread.
  /// Attempt to dequeue and resume a coroutine from head of queue.
  /// If the queue is empty, does nothing.
  ///
  /// \return `true` if a coroutine was popped and resumed; `false` if the queue is empty.
  ///
  /// \example ```cpp
  /// bool success = queue.try_pop();
  /// ```
  ///
  /// \warning After a coroutine is popped from the queue, it resumes on the queue thread, not its original thread.
  ///
  /// You can even change the manager thread, thread-safety is also controlled by `TMutex`.
  /// Note that coroutines will be resumed on the new manager thread.
  /// So, pay attention if you want to do this.
  inline bool try_pop();

  /// \brief Called by coroutines who want to suspend and wait the queue.
  /// Coroutines call `co_await queue.schedule()` to suspend, and
  /// wait for a manager thread calling `queue.try_pop()`.
  ///
  /// \example ```cpp
  /// co_await queue.schedule();
  /// ```
  ///
  /// \warning When the `queue` destruct, all the coroutines waiting the queue will be killed (destroyed).
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

  ARIA_COPY_MOVE_ABILITY(queue, delete, delete);

  /// \warning When the `queue` destruct, all the coroutines waiting the queue will be killed (destroyed).
  /// Their coroutine contexts will be destroyed and
  /// all the corresponding destructors of all the local variables will be called.
  inline ~queue() noexcept;

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
  //! If you are trying to understand the codes, please first read `Station.h` before continue.
  //!
  //! You may wonder why the template parameter is not `TQueue`.
  //! High performance concurrent queues seem to be better choices than `std::queue` with a mutex.
  //! There are 2 reasons:
  //! 1. It is almost impossible to generate safe destructors if concurrent queues are used,
  //!    because we need to loop through and destroy all the coroutines in the queue.
  //!    Within this process, other threads may simultaneously pushing in coroutines.
  //!    It is only safe enough when a mutex and the `validity` are used together.
  //! 2. Methods `try_pop()` of concurrent queues are not guaranteed to be `noexcept`,
  //!    even `tbb::concurrent_queue`.
  //!    But `std::queue` is always `noexcept`.

  std::queue<std::coroutine_handle<>> q_{};

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
#include "ARIA/Coroutine/detail/Queue.inc"
