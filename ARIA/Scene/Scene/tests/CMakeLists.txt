if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Scene.Scene
      SOURCES
      Components/Transform.cpp
      Component.cpp
      Object.cpp
      HARD_DEPENDENCIES
      ARIA::Scene::Scene
  )
endif ()
