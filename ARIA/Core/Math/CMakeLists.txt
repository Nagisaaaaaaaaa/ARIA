aria_setup_module(Core Math)

CPMAddPackage(
    NAME Eigen
    VERSION 3.4.0
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz

    # Eigen's CMakelists are not intended for library use
    DOWNLOAD_ONLY YES
)

if (Eigen_ADDED)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen3::Eigen INTERFACE ${Eigen_SOURCE_DIR})

  # This is for other libraries to identify the provided Eigen instead of the bundled one
  set(EIGEN3_INCLUDE_DIR "${Eigen_SOURCE_DIR}/Eigen/Eigen")
endif ()

set(core_math_headers
    ARIA/BezierCurve.h
    ARIA/Mat.h
    ARIA/Math.h
    ARIA/MaxwellBoltzmannDistribution.h
    ARIA/MortonCode.h
    ARIA/MovingPoint.h
    ARIA/Quat.h
    ARIA/Vec.h
)

set(core_math_sources
)

aria_add_module(
    Core Math
    HEADERS ${core_math_headers}
    SOURCES ${core_math_sources}
    HARD_DEPENDENCIES
    ARIA::Core::Core
    Eigen3::Eigen
    CMAKE_SUBDIRS tests
)
