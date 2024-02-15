set(_aria_core_modules
    Concurrency
    Core
    Coroutine
    Geometry
    Math
    SanitizerGTestMain
)

include(ARIA_AddLibrary)
include(ARIA_AddModule)
include(ARIA_AddTests)
include(ARIA_LinkModuleToLibrary)
include(ARIA_SetupModule)

aria_add_library(Core)

# Remove the "lib" prefix from shared libraries.
set(CMAKE_SHARED_LIBRARY_PREFIX_CXX "")

foreach (module ${_aria_core_modules})
  add_subdirectory(${module})
  aria_link_module_to_library(Core ${module})
  add_dependencies(ARIA-Core ARIA::Core::${module})
endforeach ()

# Export the list of modules to the parent scope.
set(_aria_core_modules ${_aria_core_modules} PARENT_SCOPE)
