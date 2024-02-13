aria_setup_module(Core Coroutine)

CPMAddPackage(
    NAME cppcoro
    GIT_TAG main
    GITHUB_REPOSITORY andreasbuhr/cppcoro
)

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
