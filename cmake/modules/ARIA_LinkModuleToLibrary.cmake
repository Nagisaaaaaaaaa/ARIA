function(aria_link_module_to_library library_name module_name)
  target_link_libraries(ARIA-${library_name} PUBLIC ARIA::${library_name}::${module_name})
endfunction()
