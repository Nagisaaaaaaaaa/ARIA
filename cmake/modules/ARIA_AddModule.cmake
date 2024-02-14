# A general function to add a module to the project.
# An example would be:
# ```cmake
# aria_add_module(
# Core Coro
# HEADERS ${core_coro_headers}
# SOURCES ${core_coro_sources}
# HARD_DEPENDENCIES
# ARIA::Core::Core
# cppcoro
# TBB::tbb
# SOFT_DEPENDENCIES
# ARIA_ENABLE_CUDA ARIA::...
# ARIA_ENABLE_CUDA ARIA::...
# CMAKE_SUBDIRS tests
# ENABLE_IF ARIA_BUILD_EXAMPLES
# )
# ```
function(aria_add_module project_name module_name)
  set(options STATIC DYNAMIC EXECUTABLE)
  set(one_value_args "")
  set(multi_value_args HEADERS SOURCES HARD_DEPENDENCIES SOFT_DEPENDENCIES CMAKE_SUBDIRS ENABLE_IF)
  cmake_parse_arguments(MODULE "${options}" "${one_value_args}" "${multi_value_args}" "${ARGN}")

  include(ARIA_Message)
  # If the module is not defined or enabled, return early.
  foreach (enable_option ${MODULE_ENABLE_IF})
    if (NOT DEFINED ${enable_option} OR NOT ${enable_option})
      aria_message(AUTHOR_WARNING "ARIA:    Ignoring module ${BoldCyan}ARIA::${project_name}::${module_name}${ColorReset} due to required option ${enable_option} is not defined or evaluated to be TRUE.")
    endif ()
  endforeach ()

  set(HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set(SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")

  list(TRANSFORM MODULE_HEADERS PREPEND ${HEADER_ROOT}/ OUTPUT_VARIABLE headers)
  list(TRANSFORM MODULE_SOURCES PREPEND ${SOURCE_ROOT}/ OUTPUT_VARIABLE sources)

  # It is complex to handle interface modules on Windows.
  if (NOT DEFINED MODULE_SOURCES)
    list(APPEND MODULE_SOURCES "${ARIA_SOURCE_DIR}/ARIA/Core/Core/_.cpp")
    list(APPEND sources "${ARIA_SOURCE_DIR}/ARIA/Core/Core/_.cpp")
  endif ()

  aria_message(STATUS "ARIA:  Adding module: ${BoldCyan}ARIA::${project_name}::${module_name}${ColorReset}")

  if (NOT MODULE_EXECUTABLE)
    if (DEFINED MODULE_SOURCES)
      set(module_link_type PUBLIC)

      if (MODULE_STATIC)
        set(module_library_type STATIC)
      elseif (MODULE_DYNAMIC)
        set(module_library_type SHARED)
      else ()
        set(module_library_type SHARED)
      endif ()
    else ()
      set(module_link_type INTERFACE)
      set(module_library_type INTERFACE)
    endif ()
  else ()
    # If it is executable module, set only module_link_type.
    set(module_link_type PUBLIC)
  endif ()

  if (NOT MODULE_EXECUTABLE)
    aria_message(STATUS "ARIA:    module library type: ${module_library_type}")
  else ()
    aria_message(STATUS "ARIA:    module executable: ${BoldYellow}true${ColorReset}")
  endif ()

  # Filter the soft dependencies.
  set(module_soft_dependencies)

  if (MODULE_SOFT_DEPENDENCIES)
    list(LENGTH MODULE_SOFT_DEPENDENCIES _length)
    math(EXPR _length "${_length}-1")

    foreach (_i RANGE 0 ${_length} 2)
      list(GET MODULE_SOFT_DEPENDENCIES ${_i} _option)
      math(EXPR _j "${_i}+1")
      list(GET MODULE_SOFT_DEPENDENCIES ${_j} _dependency)

      if (${${_option}})
        list(APPEND module_soft_dependencies ${_dependency})
      endif ()
    endforeach ()
  endif ()

  if (MODULE_EXECUTABLE)
    add_executable(ARIA-${project_name}-${module_name} ${headers} ${sources})
  else ()
    add_library(ARIA-${project_name}-${module_name}
        ${module_library_type} # STATIC, SHARED, or INTERFACE.
        ${headers}
        ${sources}
    )
    add_library(ARIA::${project_name}::${module_name} ALIAS ARIA-${project_name}-${module_name})
  endif ()

  target_include_directories(ARIA-${project_name}-${module_name}
      ${module_link_type}
      $<BUILD_INTERFACE:${HEADER_ROOT}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(ARIA-${project_name}-${module_name}
      ${module_link_type}
      ${MODULE_HARD_DEPENDENCIES} # Add the hard dependencies as link dep.
      ${module_soft_dependencies} # Filter and add the soft dependencies as link dep.
  )

  # Add hard dependencies.
  foreach (dep ${MODULE_HARD_DEPENDENCIES})
    add_dependencies(ARIA-${project_name}-${module_name} ${dep})
  endforeach ()

  # Add soft dependencies.
  foreach (dep ${module_soft_dependencies})
    add_dependencies(ARIA-${project_name}-${module_name} ${dep})
  endforeach ()

  # Add subdirectories.
  foreach (subdir ${MODULE_CMAKE_SUBDIRS})
    add_subdirectory(${subdir})
  endforeach ()

  # Enable clang-tidy on this generated target.
  include(ARIA_EnableClangTidy)
  aria_enable_clang_tidy(ARIA-${project_name}-${module_name})

  # For Windows.
  if (WIN32)
    set_target_properties(ARIA-${project_name}-${module_name} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)
  endif ()
endfunction()
