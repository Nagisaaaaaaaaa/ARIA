if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Core.Math
      SOURCES
      BezierCurve.cu
      Mat.cu
      Math.cpp
      MaxwellBoltzmannDistribution.cpp
      MortonCode.cpp
      MovingPoint.cpp
      Quat.cpp
      Vec.cpp
      HARD_DEPENDENCIES
      ARIA::Core::Math
  )
endif ()
