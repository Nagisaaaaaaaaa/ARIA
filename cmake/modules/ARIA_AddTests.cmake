# A general function to add a test for ARIA.
#
# This cmake function is a subset of ARIA_AddModule for now.
# Note that it will automatically link gtest and SanitizerGTestMain
# against the test target for simplicity.
#
# Developers should name the test like `Tests.ARIA.Core` to avoid name conflicts.
function(aria_add_tests tests_name)
  if (NOT ARIA_BUILD_TESTS)
    message(FATAL_ERROR "ARIA:    Attempted to add test ${tests_name} but ARIA_BUILD_TESTS is not set")
  endif ()

  set(options INDEPENDENT)
  set(one_value_args ENABLE_IF)
  # Only support this arg for now.
  set(multi_value_args SOURCES HARD_DEPENDENCIES)

  cmake_parse_arguments(TEST "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  # If tests are not enabled, return early.
  if (DEFINED TEST_ENABLE_IF)
    if (NOT ${TEST_ENABLE_IF}})
      return()
    endif ()
  endif ()

  aria_message(STATUS "ARIA:    module tests name: ${BoldGreen}${tests_name}${ColorReset}")

  set(test_dependencies gtest)

  if (NOT TEST_INDEPENDENT)
    list(APPEND test_dependencies ARIA::Core::SanitizerGTestMain)
  endif ()

  add_executable(${tests_name} ${TEST_SOURCES})
  target_link_libraries(${tests_name}
      PUBLIC
      ${test_dependencies}
      ${TEST_HARD_DEPENDENCIES} # Add the hard dependencies to the test
  )

  # Export all tests to a single test directory so that we can access the test conveniently.
  set_target_properties(${tests_name} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

  if (WIN32)
    add_custom_command(TARGET ${tests_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${tests_name}> $<TARGET_FILE_DIR:${tests_name}>
        COMMAND_EXPAND_LISTS)

    # # Get the runtime dlls.
    # set(runtime_dlls "$<TARGET_RUNTIME_DLLS:${tests_name}>")
    # # Create a file to save the dll lists.
    # set(dll_list_file "${CMAKE_CURRENT_BINARY_DIR}/${tests_name}.RuntimeDLLs.txt")
    # # Write the dlls to the file.
    # file(GENERATE OUTPUT "${dll_list_file}" CONTENT "${runtime_dlls}")
    #
    # # Test and copy dlls.
    # if (NOT "${runtime_dlls}" STREQUAL "")
    #   add_custom_command(TARGET ${tests_name} POST_BUILD
    #       COMMAND ${CMAKE_COMMAND} -P "${CMAKE_SOURCE_DIR}/cmake/scripts/ARIA_TestAndCopyDLLs.cmake" "${tests_name}" "${dll_list_file}" "${CMAKE_COMMAND}" "$<TARGET_FILE_DIR:${tests_name}>"
    #       COMMAND_EXPAND_LISTS
    #   )
    # endif ()

    add_definitions("-D_USE_MATH_DEFINES")
  endif ()
endfunction()
