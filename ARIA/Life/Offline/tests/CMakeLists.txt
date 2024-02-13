if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Life.Offline
      SOURCES
      Destroy.cpp
      HARD_DEPENDENCIES
      ARIA::Life::Offline
  )

  add_subdirectory(Lifecycle)
endif ()
