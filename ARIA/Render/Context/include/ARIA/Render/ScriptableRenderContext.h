#pragma once

#include "ARIA/ARIA.h"

namespace ARIA {

/// \brief Defines state and drawing commands that custom render pipelines use.
///
/// When you define a custom `RenderPipeline`, you use a `ScriptableRenderContext` to
/// schedule and submit state updates and drawing commands to the GPU.
///
/// A `RenderPipeline.Render` method implementation typically
/// culls objects that the render pipeline doesn't need to render for every Camera (see CullingResults),
/// and then makes a series of calls to
/// `ScriptableRenderContext.DrawRenderers` intermixed with `ScriptableRenderContext.ExecuteCommandBuffer` calls.
/// These calls set up global Shader properties, change render targets, dispatch compute shaders, and
/// other rendering tasks.
/// To actually execute the render loop, call `ScriptableRenderContext.Submit`.
class ScriptableRenderContext final {
public:
  void DrawRenderers();

public:
  ScriptableRenderContext() = default;

  ScriptableRenderContext(const ScriptableRenderContext &) = delete;
  ScriptableRenderContext(ScriptableRenderContext &&) noexcept = delete;
  ScriptableRenderContext &operator=(const ScriptableRenderContext &) = delete;
  ScriptableRenderContext &operator=(ScriptableRenderContext &&) noexcept = delete;
  ~ScriptableRenderContext() = default;
};

} // namespace ARIA
