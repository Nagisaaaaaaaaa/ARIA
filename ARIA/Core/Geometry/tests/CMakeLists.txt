if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Core.Geometry
      SOURCES
      AABB.cpp
      BVH.cu
      CollisionDetection.cpp
      MarchingCube.cu
      Triangle.cpp
      VDB.cu
      HARD_DEPENDENCIES
      ARIA::Core::Geometry
  )
endif ()
