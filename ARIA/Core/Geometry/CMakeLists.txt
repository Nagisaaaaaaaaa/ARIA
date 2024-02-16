aria_setup_module(Core Geometry)

set(core_geometry_headers
    ARIA/AABB.h
)

set(core_geometry_sources
)

aria_add_module(
    Core Geometry
    HEADERS ${core_geometry_headers}
    SOURCES ${core_geometry_sources}
    HARD_DEPENDENCIES
    ARIA::Core::Math
    CMAKE_SUBDIRS tests
)
