function(aria_add_library library_name)
  add_library(ARIA-${library_name} STATIC ${CMAKE_SOURCE_DIR}/ARIA/Core/Core/_.cpp)
  add_library(ARIA::${library_name} ALIAS ARIA-${library_name})
endfunction()
