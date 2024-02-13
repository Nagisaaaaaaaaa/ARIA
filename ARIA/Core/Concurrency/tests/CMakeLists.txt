if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Core.Concurrency
      SOURCES
      SpinLock.cpp
      HARD_DEPENDENCIES
      ARIA::Core::Concurrency
  )
endif ()
