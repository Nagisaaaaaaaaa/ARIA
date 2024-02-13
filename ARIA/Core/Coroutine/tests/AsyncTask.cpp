#include "ARIA/Coroutine/AsyncTask.h"

#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <gtest/gtest.h>

namespace ARIA {

namespace {

static cppcoro::static_thread_pool threadPool;
static std::atomic<int> a;
static std::atomic<int> t;

Coroutine::async_task AsyncTask() {
  ++a;
  co_await threadPool.schedule();
  ++a;
  co_await threadPool.schedule();
  ++a;
}

cppcoro::task<> Task() {
  ++t;
  co_await AsyncTask();
  ++t;
  Coroutine::async_task a0 = AsyncTask();
  Coroutine::async_task a1 = std::move(a1);
  co_await a0;
  co_await a1;
  ++t;
}

void Main() {
  cppcoro::sync_wait(Task());

  using namespace std::chrono_literals;
  std::this_thread::sleep_for(0.5s);

  EXPECT_EQ(t, 3);
  EXPECT_EQ(a, 6);
}

} // namespace

TEST(Fragment, Base) {
  try {
    Main();
  } catch (std::exception &e) { std::cout << e.what() << std::endl; }
}

} // namespace ARIA
