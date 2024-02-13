aria_setup_module(Render Pipeline)

set(render_pipeline_headers
    ARIA/Render/DefaultRenderPipeline.h
    ARIA/Render/RenderPipeline.h
    ARIA/Render/RenderPipelineManager.h
)

set(render_pipeline_sources
    DefaultRenderPipeline.cpp
    RenderPipelineManager.cpp
)

aria_add_module(
    Render Pipeline
    HEADERS ${render_pipeline_headers}
    SOURCES ${render_pipeline_sources}
    HARD_DEPENDENCIES
    ARIA::Render::Context
)
