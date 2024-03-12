if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Core.Geometry
      SOURCES
      AABB.cpp
      BVH.cu
      VDB.cu
      HARD_DEPENDENCIES
      ARIA::Core::Geometry
  )
endif ()
