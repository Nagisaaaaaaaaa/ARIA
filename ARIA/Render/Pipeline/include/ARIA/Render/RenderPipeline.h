#pragma once

#include "ARIA/Render/ScriptableRenderContext.h"

namespace ARIA {

class RenderPipeline {
public:
  virtual void Render(ScriptableRenderContext &context) = 0;

public:
  RenderPipeline() = default;

  RenderPipeline(const RenderPipeline &) = delete;
  RenderPipeline(RenderPipeline &&) noexcept = delete;
  RenderPipeline &operator=(const RenderPipeline &) = delete;
  RenderPipeline &operator=(RenderPipeline &&) noexcept = delete;
  virtual ~RenderPipeline() = default;
};

} // namespace ARIA
