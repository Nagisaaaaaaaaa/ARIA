if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Core.Coroutine
      SOURCES
      AsyncTask.cpp
      Queue.cpp
      Station.cpp
      SyncWait.cpp
      Task.cpp
      WhenAll.cpp
      HARD_DEPENDENCIES
      ARIA::Core::Coroutine
  )
endif ()
