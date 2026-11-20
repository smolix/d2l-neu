# Gradient Penalties and Convergence: Launch Plumbing (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses it at import time). It holds `cuda_graphed`, the CUDA-graph replay that
`train_grid` of :numref:`subsec_gan_25_gaussians` runs its training step
through. The section explains it in one sentence: one step of those small
networks launches a few hundred short kernels, and issuing them from Python
takes longer than running them (:numref:`subsec_comp-overhead`). Off CUDA
the step runs eagerly, so the notebook also runs on a machine without a GPU.

Moved here from the PyTorch cell `convergence-mode-coverage-on-25-gaussians-3`
in the 2026-11 quality pass (code-guide section 1: scaffolding is written once,
in the library, or not at all). The block is tagged `%%tab pytorch`; a
module-level import would join the `_preamble` shard and move the freshness
fingerprint of every notebook in the book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstring (the
section label: `build_lib.py` reads only `sec_*` labels). -->
:label:`sec_gan_convergence`

```{.python .input #convergence-lib-cuda-graphed}
%%tab pytorch
#@save
def cuda_graphed(step, device, warmup=3):
    """Return a function that calls step(). On a CUDA device it runs step()
    eagerly for warmup calls on a side stream, then records one call as a
    CUDA graph and replays it for every later call; elsewhere it calls step()
    directly. Replay reuses the recorded memory addresses, so the optimizers
    must keep their state and learning rate on the device (capturable=True,
    learning rate as a tensor)."""
    if torch.device(device).type != 'cuda':
        return step
    graph, calls = torch.cuda.CUDAGraph(), 0
    def run():
        nonlocal calls
        calls += 1
        if calls <= warmup:      # allocates optimizer state and workspaces
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                step()
            torch.cuda.current_stream().wait_stream(side)
            return
        if calls == warmup + 1:  # recording launches nothing
            with torch.cuda.graph(graph):
                step()
        graph.replay()
    return run
```
