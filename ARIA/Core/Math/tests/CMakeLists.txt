if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Core.Math
      SOURCES
      Mat.cpp
      Math.cpp
      Quat.cpp
      Vec.cpp
      HARD_DEPENDENCIES
      ARIA::Core::Math
  )
endif ()
