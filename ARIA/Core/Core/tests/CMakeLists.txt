if (ARIA_BUILD_TESTS)
  aria_add_tests(Tests.ARIA.Core.Core
      SOURCES
      detail/Macros.cpp
      #experimental/Lua.cpp
      Auto.cpp
      BFloat16.cu
      BitArray.cu
      BitVector.cu
      Constant.cpp
      DisjointSet.cu
      Float16.cu
      ForEach.cpp
      Invocations.cpp
      Launcher.cu
      Layout.cu
      Let.cpp
      Mosaic.cu
      Property.cpp
      PropertySTL.cpp
      Python.cpp
      Registry.cpp
      TensorVector.cu
      TypeArray.cpp
      HARD_DEPENDENCIES
      ARIA::Core::Core
  )
endif ()
