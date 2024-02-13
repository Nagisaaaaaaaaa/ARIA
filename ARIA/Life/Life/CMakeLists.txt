aria_setup_module(Life Life)

set(life_life_headers
    ARIA/Life/Lifecycle.h
)

set(life_life_sources
)

aria_add_module(
    Life Life
    HEADERS ${life_life_headers}
    SOURCES ${life_life_sources}
    HARD_DEPENDENCIES
    ARIA::Scene
)
