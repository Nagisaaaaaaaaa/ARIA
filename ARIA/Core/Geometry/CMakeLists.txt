aria_setup_module(Core Geometry)

CPMAddPackage(
    NAME stdgpu
    GITHUB_REPOSITORY stotko/stdgpu
    GIT_TAG master
    OPTIONS "STDGPU_BUILD_EXAMPLES OFF" "STDGPU_BUILD_BENCHMARKS OFF" "STDGPU_BUILD_TESTS OFF"
)

set(core_geometry_headers
    ARIA/AABB.h
    ARIA/BVH.h
    ARIA/CollisionDetection.h
    ARIA/MarchingCube.h
    ARIA/Triangle.h
    ARIA/VDB.h
)

set(core_geometry_sources
)

aria_add_module(
    Core Geometry
    HEADERS ${core_geometry_headers}
    SOURCES ${core_geometry_sources}
    HARD_DEPENDENCIES
    ARIA::Core::Math
    stdgpu::stdgpu
    CMAKE_SUBDIRS tests
)
