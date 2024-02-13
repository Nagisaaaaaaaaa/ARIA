#include "ARIA/Coroutine/SyncWait.h"
#include "ARIA/Coroutine/Task.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(SyncWait, Base) {
  auto task = []() -> Coroutine::task<> { co_return; };

  Coroutine::sync_wait(task());
}

} // namespace ARIA
