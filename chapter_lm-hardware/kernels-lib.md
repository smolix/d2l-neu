# Kernels, Tiles and Fusion: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`. It is in `LIB_ONLY_EARLY_FILES` (emitted before the
numbered chapters): the Triton kernels of :numref:`sec_lm_kernels` are shown
on the page unchanged, with `@triton.jit` and `tl.constexpr` evaluated when
`d2l.torch` is imported, so `triton` and `tl` must be bound first. On a
machine without Triton (macOS) both are bound to an inert stand-in: the
kernels are defined, and launching one raises
`RuntimeError('Triton is not installed')`. It holds no registry entry.

Later steps add 22.2's flash SDPA wrapper, sweeps, timers and displays here
(`sdpa_causal`, `hw_shape_sweep`, `hw_wave_sweep`, `flash_sweep`,
`time_attention`, `time_fns`, `attention_shapes`, `attention_inputs`,
`attention_error`, `check_close`, `triton_spills`, `inductor_kernels`,
displays; warnings silenced through `d2l._quiet`). Blocks are tagged
`%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_kernels`

## The Triton Binding

```{.python .input #kernels-lib-no-triton}
%%tab pytorch
#@save
class _TritonMissing:
    """A kernel defined without Triton: launching it raises."""
    def __getitem__(self, grid):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError('Triton is not installed')

#@save
class _NoTriton:
    """Stands in for triton and triton.language without Triton: decorators
    (jit, autotune, heuristics) give kernels that raise when launched, and
    every other attribute (tl.constexpr, tl.float32, triton.Config) is
    inert."""
    def __getattr__(self, name):
        return _NoTriton()

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _TritonMissing()  # used as a decorator
        return _NoTriton()

    def jit(self, fn=None, **kwargs):
        return _TritonMissing() if fn is not None else self

#@save
def _triton_modules():
    """(triton, triton.language), or two stand-ins without Triton."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return _NoTriton(), _NoTriton()
    return triton, tl
```

```{.python .input #kernels-lib-bind}
%%tab pytorch
#@save
triton = _triton_modules()[0]
tl = _triton_modules()[1]
```
