if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Core.Core
      SOURCES
      detail/Macros.cpp
      Auto.cpp
      Concepts.cpp
      Constant.cpp
      DisjointSet.cu
      ForEach.cpp
      Launcher.cu
      Layout.cu
      Property.cpp
      PropertySTL.cpp
      Registry.cpp
      TensorVector.cu
      TypeArray.cpp
      HARD_DEPENDENCIES
      ARIA::Core::Core
  )
endif ()
