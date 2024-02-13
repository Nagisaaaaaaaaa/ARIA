#pragma once

#include "RenderPipeline.h"

namespace ARIA {

class DefaultRenderPipeline final : public RenderPipeline {
public:
  void Render(ScriptableRenderContext &context) final;

public:
  DefaultRenderPipeline() = default;

  DefaultRenderPipeline(const DefaultRenderPipeline &) = delete;
  DefaultRenderPipeline(DefaultRenderPipeline &&) noexcept = delete;
  DefaultRenderPipeline &operator=(const DefaultRenderPipeline &) = delete;
  DefaultRenderPipeline &operator=(DefaultRenderPipeline &&) noexcept = delete;
  ~DefaultRenderPipeline() final = default;
};

} // namespace ARIA
