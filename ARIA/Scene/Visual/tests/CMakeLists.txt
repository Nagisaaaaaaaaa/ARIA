if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Scene.Visual
      SOURCES
      Camera.cpp
      HARD_DEPENDENCIES
      ARIA::Scene::Visual
  )

  add_subdirectory(Window)
endif ()
