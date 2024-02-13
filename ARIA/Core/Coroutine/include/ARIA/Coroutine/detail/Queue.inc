#pragma once

namespace ARIA::Coroutine {

//! If you are trying to understand the codes, please first read `Station.h` before continue.

template <typename TMutex>
inline bool queue<TMutex>::try_pop() {
  std::lock_guard guard{mutex_};

  if (q_.empty())
    return false;

  q_.front().resume();
  q_.pop();

  return true;
}

template <typename TMutex>
inline auto queue<TMutex>::schedule() noexcept {
  class Awaiter {
  public:
    Awaiter(queue &queue) : queue_(queue) {}

    // `co_await` a `queue` will always suspend the current coroutine and
    // return execution back to the caller because we assume that `queue` are
    // handled only by a manager thread.
    bool await_ready() const noexcept { return false; }

    // `emplace` the coroutine, which is currently calling `co_await` the `queue`, to the waiting list.
    // The coroutine will suspend and wait until the manager thread pops and resumes it.
    void await_suspend(std::coroutine_handle<> c) {
      std::lock_guard guard{queue_.mutex_};

      if (!queue_.validity()) [[unlikely]]
        ARIA_THROW(std::runtime_error, "`co_await` a destructing `queue` will result in undefined behaviors");

      queue_.q_.push(c);
    }

    // `co_await` returns void.
    void await_resume() noexcept {}

  private:
    queue &queue_;
  };

  return Awaiter{*this};
}

//
//
//
template <typename TMutex>
inline queue<TMutex>::~queue() noexcept {
  std::lock_guard guard{mutex_};

  validity() = false;

  while (!q_.empty()) {
    q_.front().destroy();
    q_.pop();
  }
}

} // namespace ARIA::Coroutine
