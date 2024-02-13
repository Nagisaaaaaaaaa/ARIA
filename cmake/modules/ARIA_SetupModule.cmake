macro(aria_setup_module project_name module_name)
  set(ARIA_MODULE_NAME ARIA-${project_name}-${module_name})
  project(${ARIA_MODULE_NAME})
endmacro()
