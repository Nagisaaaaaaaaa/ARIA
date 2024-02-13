macro(aria_setup_plugin project_name plugin_name)
  set(ARIA_PLUGIN_NAME ARIA-${project_name}-${plugin_name})
  project(${ARIA_PLUGIN_NAME})
endmacro()
