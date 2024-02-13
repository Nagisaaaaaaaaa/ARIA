set(_aria_life_modules
    Life
    Offline
    Realtime
)

include(ARIA_AddLibrary)
include(ARIA_AddModule)
include(ARIA_AddTests)
include(ARIA_LinkModuleToLibrary)
include(ARIA_SetupModule)

aria_add_library(Life)

set(CMAKE_SHARED_LIBRARY_PREFIX_CXX "")

foreach (module ${_aria_life_modules})
  add_subdirectory(${module})
  aria_link_module_to_library(Life ${module})
endforeach ()

set(_aria_life_modules ${_aria_life_modules} PARENT_SCOPE)
