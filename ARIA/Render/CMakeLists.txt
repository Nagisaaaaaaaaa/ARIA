set(_aria_render_modules
    Context
    Pipeline
)

include(ARIA_AddLibrary)
include(ARIA_AddModule)
include(ARIA_AddTests)
include(ARIA_LinkModuleToLibrary)
include(ARIA_SetupModule)

aria_add_library(Render)

set(CMAKE_SHARED_LIBRARY_PREFIX_CXX "")

foreach (module ${_aria_render_modules})
  add_subdirectory(${module})
  aria_link_module_to_library(Render ${module})
endforeach ()

set(_aria_render_modules ${_aria_render_modules} PARENT_SCOPE)
