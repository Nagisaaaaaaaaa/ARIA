# A function for enabling clang-tidy checks under a target.
# ```cmake
# aria_enable_clang_tidy(aria::core)
# ```
function(aria_enable_clang_tidy target_name)
  if(ARIA_USE_CLANG_TIDY)
    find_program(ARIA_CXX_CLANG_TIDY
      NAMES "clang-tidy"
      DOC "Path to clang-tidy executable")

    if(NOT ARIA_CXX_CLANG_TIDY)
      message(FATAL_ERROR "clang-tidy not found, but ARIA_USE_CLANG_TIDY is ON. Please install clang-tidy or disable ARIA_USE_CLANG_TIDY.")
    else()
      message(STATUS "ARIA:    Enable clang-tidy for ${target_name}")
    endif()

    set_target_properties(${target_name} PROPERTIES
      CXX_CLANG_TIDY ${ARIA_CXX_CLANG_TIDY})
  endif()
endfunction()
