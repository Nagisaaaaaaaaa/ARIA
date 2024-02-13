aria_setup_module(Core Concurrency)

set(core_concurrency_headers
    ARIA/Concurrency/SpinLock.h
)

set(core_concurrency_sources
)

aria_add_module(
    Core Concurrency
    HEADERS ${core_concurrency_headers}
    SOURCES ${core_concurrency_sources}
    HARD_DEPENDENCIES
    ARIA::Core::Core
    CMAKE_SUBDIRS tests
)
