#include "ARIA/Render/DefaultRenderPipeline.h"

namespace ARIA {

void DefaultRenderPipeline::Render(ScriptableRenderContext &context) {
  context.DrawRenderers();
}

} // namespace ARIA
