aria_setup_module(Life Offline)

set(life_offline_headers
    ARIA/Life/Offline/Destroy.h
    ARIA/Life/Offline/Lifecycle.h
    ARIA/Life/Offline/MonoBehavior.h
)

set(life_offline_sources
    Destroy.cpp
    Lifecycle.cpp
    MonoBehavior.cpp
)

aria_add_module(
    Life Offline
    HEADERS ${life_offline_headers}
    SOURCES ${life_offline_sources}
    HARD_DEPENDENCIES
    ARIA::Life::Life
    ARIA::Render
    CMAKE_SUBDIRS tests
)
