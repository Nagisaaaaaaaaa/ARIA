#include "ARIA/ARIA.h"

#include <gtest/gtest.h>

#if defined(ARIA_ENABLE_CUDA) && defined(__SANITIZE_ADDRESS__)
// GCC does not have memory sanitizer support yet, so there will be a warning.
  #if ARIA_GCC
    #define ARIA_SANITIZER_HOOK_ATTRIBUTE                                                                              \
      extern "C" __attribute__((no_sanitize("address", "thread", "undefined"))) __attribute__((visibility("default"))) \
      __attribute__((used))
  #else
    #define ARIA_SANITIZER_HOOK_ATTRIBUTE                                                                              \
      extern "C" __attribute__((no_sanitize("address", "memory", "thread", "undefined")))                              \
      __attribute__((visibility("default"))) __attribute__((used))
  #endif

// `asan` cannot work trivially under CUDA, this option is required should only be included once.
// https://github.com/nanoporetech/dorado/commit/ea47675015fdeb4e018a912cfa91c5518be2511a
ARIA_SANITIZER_HOOK_ATTRIBUTE const char *__asan_default_options() {
  return "protect_shadow_gap=0:detect_leaks=0";
}
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
