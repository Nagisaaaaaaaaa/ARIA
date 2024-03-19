#include "ARIA/Launcher.h"
#include "ARIA/VDB.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

void TestVDBHandle() {
  using Handle = VDBHandle<float, 2, SpaceDevice>;
  Handle handle = Handle::Create();

  const size_t n = 1;

  Launcher(n, [=] ARIA_DEVICE(size_t i) mutable { handle.value({i, 0}) = 233; }).Launch();

  cuda::device::current::get().synchronize();

  handle.Destroy();
}

} // namespace

TEST(VDB, Handle) {
  TestVDBHandle();
}

} // namespace ARIA
