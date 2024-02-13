#pragma once

#include "RenderPipeline.h"

#include <memory>

namespace ARIA {

class RenderPipelineManager {
public:
  static std::unique_ptr<RenderPipeline> &currentPipeline();

  static void Render();

  //
  //
  //
public:
  RenderPipelineManager() = delete;

  RenderPipelineManager(const RenderPipelineManager &) = delete;
  RenderPipelineManager(RenderPipelineManager &&) noexcept = delete;
  RenderPipelineManager &operator=(const RenderPipelineManager &) = delete;
  RenderPipelineManager &operator=(RenderPipelineManager &&) noexcept = delete;

  ~RenderPipelineManager() = delete;
};

} // namespace ARIA
