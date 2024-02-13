aria_setup_module(Life Realtime)

set(life_realtime_headers
    ARIA/Life/Realtime/Lifecycle.h
    ARIA/Life/Realtime/MonoBehavior.h
)

set(life_realtime_sources
    Lifecycle.cpp
)

aria_add_module(
    Life Realtime
    HEADERS ${life_realtime_headers}
    SOURCES ${life_realtime_sources}
    HARD_DEPENDENCIES
    ARIA::Life::Life
)
