#pragma once

namespace ARIA::Coroutine {

template <typename TMutex>
inline void station<TMutex>::go() {
  std::list<std::coroutine_handle<>> passengers;

  {
    std::lock_guard guard{mutex_};

    //! In this way, if one passenger consecutively calls `co_await` 2 times,
    //! The second call will always successfully block the passenger
    //! and will not result in a recursive dead lock.
    using std::swap;
    swap(passengers, passengers_);
  }

  for (auto &passenger : passengers)
    passenger.resume();
}

template <typename TMutex>
inline auto station<TMutex>::schedule() noexcept {
  class Awaiter {
  public:
    Awaiter(station &station) : station_(station) {}

    // `co_await` a `station` will always suspend the current coroutine and
    // return execution back to the caller because we assume that `station` are
    // handled only by the bus driver (a manager thread).
    bool await_ready() const noexcept { return false; }

    // `emplace` the coroutine, which is currently calling `co_await` the `station`, to the waiting list.
    // The coroutine will suspend and wait until the "bus driver" comes and goes.
    void await_suspend(std::coroutine_handle<> passenger) {
      std::lock_guard guard{station_.mutex_};

      //! See the comments of the destructor to understand why this `if` makes the code safer.
      if (!station_.validity()) [[unlikely]]
        ARIA_THROW(std::runtime_error, "`co_await` a destructing `station` will result in undefined behaviors");

      station_.passengers_.emplace_back(passenger);
    }

    // `co_await` returns nothing.
    void await_resume() noexcept {}

  private:
    station &station_;
  };

  return Awaiter{*this};
}

//
//
//
template <typename TMutex>
inline station<TMutex>::~station() noexcept {
  //! Q1: Should destructor also ensure thread safety?
  //! It MUST!
  //! Consider the following example:
  //!   1. Another thread is trying to `co_await` the `station` while the destructor is executing.
  //!   2. If the lock is not acquired, the following loop is non thread-safe.
  //!   3. We cannot assume all the passengers will be safely destroyed.
  //!      There may exist memory leakage.
  std::lock_guard guard{mutex_};

  //! Q2: Why simply writing the above line is not enough?
  //! Consider the following example:
  //!   1. Another thread is trying to `co_await` the `station` while the destructor is executing.
  //!   2. The destructor firstly acquires the lock and the `station` is destroyed.
  //!   3. `co_await` secondly tries to acquire the lock and add a new passenger.
  //!   4. Since destructor of the `station` has already been called, undefined behavior occurs.
  //!
  //! To prevent such weird undefined behaviors, when the destructor is called,
  //! the `station` is set to invalid and no passenger is allowed to enter.
  //!
  //! Read the comments of `SchedulerBase` to understand why a base class is introduced.
  //! Also, `validity` is implemented with the ARIA property system, see `Property.h`.
  validity() = false;

  for (auto &passenger : passengers_)
    passenger.destroy();
}

} // namespace ARIA::Coroutine
