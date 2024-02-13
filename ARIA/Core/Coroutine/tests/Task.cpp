#include "ARIA/Coroutine/Task.h"

#include <cppcoro/sync_wait.hpp>
#include <gtest/gtest.h>

namespace ARIA {

TEST(Task, Base) {
  auto task = []() -> Coroutine::task<> { co_return; };

  cppcoro::sync_wait(task());
}

} // namespace ARIA
