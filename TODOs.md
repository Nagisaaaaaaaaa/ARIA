# TODOs

Add `std::array, std::span` for tensor vector `operator()` and `operator[]`.
Add `array()`, `span()` for Eigen and other core types.

**`Layout`s are descriptions for DENSE volumes with arbitrary sizes.**
**`Layout`s have shapes defined by the DENSE volumes.**
*Arbitrary* means that Morton codes are not `Layout`s because it fails for size = (8, 9).
It is best to directly use CuTe instead of writing something.

**`Tensor`s are DENSE volumes with a `Layout` and conditionally a `Engine`.**
**`Tensor`s have shapes defined by the DENSE volumes.**
It is best to directly use CuTe instead of writing something.

**`DynLayout`s are `Layout`s with dynamic shapes at all dimensions.**
It should be resizable like an AABB volume.

**`DynTensor`s are OWNED `Tensor`s defined by `DynLayout`s with dynamic shapes at all dimensions.**
It should be resizable like an AABB volume.
The simplest LBM runs on `DynTensor`s.

**`VDB`s are SPARSE volumes like OpenVDB.**
**`VDB`s do not have shapes.**
`VDB` should support parallel modification.

Should name `TriMesh` instead of `Mesh`, because there will exist mesh refinements.

Add abstraction for mesh: `class TriMesh`, should be placed at `ARIA::Core::Geometry`.

Add abstraction for mesh filter: `class TriMeshFilter : public Component`.

Add abstraction for rendering: the `Rendering` module, which contains:
`RendererList`, `RenderPipeline`, `RenderPipeline::Render()`

Add macros for the copy and swap idioms.

Learn and use `fmt` and `spdlog`.

How to make destructors of `Station` and `Queue` noexcept?

How to abstract `OpenGL` and `Vulkan`? Together with `Window`?

If a parent `Transform` is set to dirty, all its children `Transform`s should be set to dirty, so expensive.

Add warning: never use `eval()` as member functions, or there will be undefined behaviors.

Add warning: `Registry` only works when all the codes are placed in one .cpp.

Test whether the current `Registry` support complex inheritance?

Document this: `Object` should not be organized with `Registry` because
it highly relies on a tree hierarchy.
So, the `class Object` should own something like a `static Object root`.
Then, we are able to implement `Object.Find(std::string_view)`.

How to implement a good dirty and update system?

Add more `ForEach`-like functions to `Registry`.
**Also make sure thread safety.**

Refine coding standards: Never call `std::swap`.

Disable `clang-tidy` for checking `if {}`.

The magic instructions in `SpinLock` does not support ARM, more macros needed.

`Transform` should be a `Component` class, not a `Math` class.

Support `Transform`.
