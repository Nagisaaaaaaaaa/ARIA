aria_setup_module(Core Coroutine)

CPMAddPackage(
    NAME cppcoro
    GIT_TAG main
    GITHUB_REPOSITORY andreasbuhr/cppcoro
    # The provided cmake configuration does not work with NVCC.
    DOWNLOAD_ONLY YES
)

# Manually get the headers and sources.
file(GLOB cppcoro_HEADERS "${cppcoro_SOURCE_DIR}/include/cppcoro/*.hpp")
file(GLOB cppcoro_SOURCES "${cppcoro_SOURCE_DIR}/lib/*.cpp")

# Exclude `linux.cpp` on Windows and `win32.cpp` on Linux.
if (MSVC)
  list(REMOVE_ITEM cppcoro_SOURCES "${cppcoro_SOURCE_DIR}/lib/linux.cpp")
else ()
  list(REMOVE_ITEM cppcoro_SOURCES
      "${cppcoro_SOURCE_DIR}/lib/win32.cpp"
      "${cppcoro_SOURCE_DIR}/lib/socket.cpp"
      "${cppcoro_SOURCE_DIR}/lib/socket_accept_operation.cpp"
      "${cppcoro_SOURCE_DIR}/lib/socket_connect_operation.cpp"
      "${cppcoro_SOURCE_DIR}/lib/socket_disconnect_operation.cpp"
      "${cppcoro_SOURCE_DIR}/lib/socket_recv_from_operation.cpp"
      "${cppcoro_SOURCE_DIR}/lib/socket_recv_operation.cpp"
      "${cppcoro_SOURCE_DIR}/lib/socket_send_operation.cpp"
      "${cppcoro_SOURCE_DIR}/lib/socket_send_to_operation.cpp")
endif ()

# Add and setup library.
add_library(cppcoro STATIC ${cppcoro_HEADERS} ${cppcoro_SOURCES})
target_include_directories(cppcoro PUBLIC "${cppcoro_SOURCE_DIR}/include")
if (MSVC)
  target_compile_options(cppcoro PRIVATE /std:c++latest)
  target_link_libraries(cppcoro PUBLIC synchronization.lib)
  #  target_compile_options(cppcoro PUBLIC /await) # Does not work with NVCC.
else ()
  target_compile_options(cppcoro PRIVATE -std=c++20)
endif ()

set(core_coroutine_headers
    ARIA/Coroutine/AsyncTask.h
    ARIA/Coroutine/Queue.h
    ARIA/Coroutine/Station.h
    ARIA/Coroutine/SyncWait.h
    ARIA/Coroutine/Task.h
    ARIA/Coroutine/WhenAll.h
)

set(core_coroutine_sources
)

aria_add_module(
    Core Coroutine
    HEADERS ${core_coroutine_headers}
    SOURCES ${core_coroutine_sources}
    HARD_DEPENDENCIES
    ARIA::Core::Core
    cppcoro
    CMAKE_SUBDIRS tests
)
