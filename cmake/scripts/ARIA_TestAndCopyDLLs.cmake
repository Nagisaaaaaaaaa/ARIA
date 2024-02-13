set(target ${CMAKE_ARGV3})
set(dll_list_file ${CMAKE_ARGV4})
set(cmake_command ${CMAKE_ARGV5})
set(target_dir ${CMAKE_ARGV6})

# Read the dll list from file.
file(READ "${dll_list_file}" runtime_dlls)

# If the dll list is not empty, copy all the dlls if different.
if (NOT "${runtime_dlls}" STREQUAL "")
  execute_process(COMMAND ${cmake_command} -E copy_if_different ${runtime_dlls} ${target_dir})
endif ()
