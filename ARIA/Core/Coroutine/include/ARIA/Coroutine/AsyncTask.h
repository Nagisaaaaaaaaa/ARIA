#pragma once

/// \file
/// \brief An `async_task` is a fully asynchronized coroutine,
/// which cannot be synchronized or waited.
/// `co_await` will never suspend, and `sync_wait()` will never block the execution.
///
/// Thanks to this feature, `async_task`s can be terminated in the halfway
/// without concerning about synchronization.
/// For example, when the schedulers destruct, they must destroy
/// all the coroutines contained in them.
/// It is ALWAYS DANGEROUS to destroy a `task`, but safer to destroy an `async_task`.
///
/// If you are not familiar with C++20 coroutine, read the following contexts before continue:
/// 1. A good tutorial: https://www.scs.stanford.edu/~dm/blog/c++-coroutines.html.
/// 2. A good library: https://github.com/lewissbaker/cppcoro.

//
//
//
//
//
#include "ARIA/ARIA.h"

#include <coroutine>

namespace ARIA::Coroutine {

/// \brief An `async_task` is a fully asynchronized coroutine,
/// which cannot be synchronized or waited.
///
/// An `async_task` represents fully asynchronized sequences of codes, which
/// can be easily suspended, scheduled, or terminated in the halfway.
///
/// Similar to `task`, execution of a `async_task` does not start until it is `await`ed.
///
/// \warning `async_task`s cannot be synchronized or waited, which means that
/// `co_await` will never suspend, and `sync_wait()` will never block the execution.
/// See the following example.
///
/// Thanks to this feature, `async_task`s can be terminated in the halfway
/// without concerning about synchronization.
/// For example, when the schedulers destruct, they must destroy
/// all the coroutines contained in them.
/// It is ALWAYS DANGEROUS to destroy a `task`, but safer to destroy an `async_task`.
///
/// \example ```cpp
/// task<> FTask() {
///   printf("This line is executed with thread-A");
///
///   // This line will never be suspended.
///   co_await FAsyncTask();
///
///   printf("This line is executed with thread-A");
/// }
///
/// async_task FAsyncTask() {
///   printf("This line is executed with thread-A");
///
///   // When this line is called, only `FAsyncTask` is suspended and
///   // will be scheduled by the thread pool.
///   // Then, `FTask` continues as if `FAsyncTask` has finished and returned.
///   co_await threadPool.schedule();
///
///   // After the thread pool schedules `FAsyncTask`, this line is called.
///   // So, it is likely that this line is executed with another thread.
///   printf("This line is (likely) executed with thread-B");
/// }
///
/// // WARNING, it is possible that `FTask` is finished but `FAsyncTask` is still executing, since
/// // `FAsyncTask` represents asynchronized coroutines.
/// // `sync_wait` will only wait for `FTask` and will not wait for `FAsyncTask`.
/// sync_wait(FTask());
/// ```
class async_task {
public:
  struct promise_type {
    async_task get_return_object() noexcept {
      return async_task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    // Similar to `task`s, `async_task`s also do not start until they are `await`ed.
    std::suspend_always initial_suspend() noexcept { return {}; }

    // Never suspend after finish, so that `destroy()` will be automatically called.
    std::suspend_never final_suspend() noexcept { return {}; }

    void unhandled_exception() noexcept {}

    //! `async_task`s should not have return values, because
    //! they are designed to allow safe halfway termination.
    void return_void() noexcept {}
  };

public:
  // `co_await` will resume the coroutine and will never suspend.
  std::suspend_never operator co_await() {
    if (coroutine_) {
      //! This line means that, when an `async_task` is suspended, for example if
      //! an `async_task` calls `co_await threadPool.schedule()`,
      //! it will be truly suspended and scheduled.
      //!
      //! But, execution returns back to this line and calls `return std::suspend_never{};`.
      //! This means that an `async_task` can only block itself but will never block the caller.
      //! So, if the caller is a `task`, that `task` will never be blocked and
      //! will continue as of the `async_task` is a "fully asynchronized coroutine".
      coroutine_.resume();
    }

    return {};
  }

private:
  explicit async_task(std::coroutine_handle<promise_type> coroutine) : coroutine_(coroutine) {}

public:
  ARIA_COPY_ABILITY(async_task, delete);

  async_task(async_task &&others) noexcept : coroutine_(others.coroutine_) { others.coroutine_ = nullptr; }

  async_task &operator=(async_task &&other) noexcept {
    if (std::addressof(other) != this) {
      if (coroutine_) {
        coroutine_.destroy();
      }

      coroutine_ = other.coroutine_;
      other.coroutine_ = nullptr;
    }

    return *this;
  }

  ~async_task() = default;

private:
  std::coroutine_handle<promise_type> coroutine_{};
};

} // namespace ARIA::Coroutine
