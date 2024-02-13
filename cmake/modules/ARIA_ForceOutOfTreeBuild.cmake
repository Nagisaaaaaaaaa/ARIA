# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(aria_force_out_of_tree_build)
  string(COMPARE EQUAL "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" insource)
  get_filename_component(parentdir ${PROJECT_SOURCE_DIR} PATH)
  string(COMPARE EQUAL "${PROJECT_SOURCE_DIR}" "${parentdir}" insourcesubdir)
  if(insource OR insourcesubdir)
    message(FATAL_ERROR "In-tree builds are not permitted. To recover, delete "
      "'CMakeCache.txt', the 'CMakeFiles' directory and inform CMake about "
      "the source (-S) and build (-B) paths. For example to compile to a "
      "directory labeled 'build', enter\n"
      "  $ rm -Rf CMakeCache.txt CMakeFiles\n"
      "  $ cmake -S . -B build\n"
      "  $ cmake --build build")
  endif()
endfunction()
