set(_aria_scene_modules
    Scene
    Visual
)

include(ARIA_AddLibrary)
include(ARIA_AddModule)
include(ARIA_AddTests)
include(ARIA_LinkModuleToLibrary)
include(ARIA_SetupModule)

aria_add_library(Scene)

set(CMAKE_SHARED_LIBRARY_PREFIX_CXX "")

foreach (module ${_aria_scene_modules})
  add_subdirectory(${module})
  aria_link_module_to_library(Scene ${module})
endforeach ()

set(_aria_scene_modules ${_aria_scene_modules} PARENT_SCOPE)
