aria_setup_module(Render Context)

set(render_context_headers
    ARIA/Render/ScriptableRenderContext.h
)

set(render_context_sources
    ScriptableRenderContext.cpp
)

aria_add_module(
    Render Context
    HEADERS ${render_context_headers}
    SOURCES ${render_context_sources}
    HARD_DEPENDENCIES
    ARIA::Scene
)
