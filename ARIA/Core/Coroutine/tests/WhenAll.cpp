#include "ARIA/Coroutine/WhenAll.h"
#include "ARIA/Coroutine/SyncWait.h"
#include "ARIA/Coroutine/Task.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(WhenAll, Base) {
  auto task = []() -> Coroutine::task<> { co_return; };

  std::vector<Coroutine::task<>> tasks;
  for (int i = 0; i < 10; ++i)
    tasks.emplace_back(task());

  Coroutine::sync_wait(Coroutine::when_all(std::move(tasks)));
}

} // namespace ARIA
