#include "ARIA/Render/RenderPipelineManager.h"

namespace ARIA {

static std::unique_ptr<RenderPipeline> currentPipeline_;

//
//
//
std::unique_ptr<RenderPipeline> &RenderPipelineManager::currentPipeline() {
  return currentPipeline_;
}

void RenderPipelineManager::Render() {
  ScriptableRenderContext context;

  currentPipeline()->Render(context);
}

} // namespace ARIA
