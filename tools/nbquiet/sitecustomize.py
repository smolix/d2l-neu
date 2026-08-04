"""Book-output hygiene for notebook execution: no ANSI, no progress bars, no paths.

This module is placed on ``PYTHONPATH`` by
:func:`tools.runtime_env.setup_framework_env`, so CPython's ``site`` module
imports it at interpreter startup in *every* process that executes book
content — the ``jupyter nbconvert`` parent, the ``ipykernel`` child that
actually runs the cells, and the kernel processes used for slides.

Why it exists
-------------
Book output is rendered into HTML *and* into a LaTeX ``verbatim`` block. Both
media are plain text, so anything a library writes for a *live terminal* —
ANSI colour, cursor motion, carriage returns, backspace-erased progress bars —
is captured verbatim and shows up as garbage in the PDF. Most of that noise is
switched off by environment variables (see ``QUIET_ENV`` in
``tools/runtime_env.py``). What is left here is the residue that no environment
variable can reach:

* ``flax.nnx`` decides on colour from "``sys.stdout`` is a tty *or* IPython is
  importable" and ignores ``NO_COLOR`` entirely, so every ``nnx.state(...)``
  repr in a notebook comes out full of 24-bit colour escapes.
* ``rich`` (used by ``keras.Model.summary()``) hardcodes TRUECOLOR whenever it
  detects a Jupyter kernel, again ignoring ``NO_COLOR``/``TERM``.
* Python's default warning format prints the absolute path of the file that
  raised — which in a book means a machine-specific
  ``/home/<user>/…/.venv-jax/lib/python3.12/site-packages/…`` line, or a
  throwaway ``/tmp/ipykernel_65430/1272845916.py``.

Everything here is *presentation only*: it never changes a computed value, and
it never silences a warning a chapter deliberately demonstrates (see
``_IGNORED_WARNINGS`` — a short, explicit list of third-party deprecation
chatter, not a blanket filter).

Set ``D2L_NO_QUIET=1`` to disable, e.g. when debugging a library's own output.
"""

import os
import sys

# Keep array/frame reprs inside the PDF's ~80-column verbatim budget, and keep
# warnings inside it too.
_WRAP_WIDTH = 80

_HOOKS = {}


class _PostImportFinder:
    """Run ``_HOOKS[name](module)`` right after ``name`` finishes importing.

    ``sitecustomize`` runs before a notebook imports anything, so a library can
    only be adjusted *after* it loads. This finder sits at the head of
    ``sys.meta_path``, delegates the actual search to the real finders, then
    wraps the returned loader's ``exec_module`` with our callback.
    """

    def __init__(self):
        self._busy = set()

    def find_spec(self, fullname, path=None, target=None):
        fn = _HOOKS.get(fullname)
        if fn is None or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                try:
                    spec = find(fullname, path, target)
                except Exception:
                    spec = None
                if spec is not None:
                    break
        finally:
            self._busy.discard(fullname)
        loader = getattr(spec, "loader", None)
        if spec is None or loader is None or not hasattr(loader, "exec_module"):
            return spec
        original = loader.exec_module

        def exec_module(module, _original=original, _fn=fn):
            _original(module)
            try:
                _fn(module)
            except Exception:  # never let cosmetics break an import
                pass

        try:
            loader.exec_module = exec_module
        except Exception:
            pass
        return spec


def _on_import(name, fn):
    _HOOKS[name] = fn
    mod = sys.modules.get(name)
    if mod is not None:  # already imported (only if something pre-imports it)
        try:
            fn(mod)
        except Exception:
            pass


# ── flax.nnx: strip 24-bit colour from Pytree/State reprs ───────────────
# flax.nnx.reprlib.supports_color() returns True whenever IPython is importable,
# which is always true inside a kernel; there is no env var. Rebind the module's
# palette to its own all-empty NO_COLOR palette — on the module, on the
# thread-local context class, on its live instance, and on the dataclass
# __init__ default (which threading.local re-applies for every new thread).
def _quiet_flax_nnx(reprlib):
    plain = reprlib.NO_COLOR
    reprlib.COLOR = plain
    ctx_cls = getattr(reprlib, "ReprContext", None)
    if ctx_cls is not None:
        ctx_cls.current_color = plain
        init = getattr(ctx_cls, "__init__", None)
        defaults = getattr(init, "__defaults__", None)
        if defaults:
            init.__defaults__ = tuple(
                plain if isinstance(d, type(plain)) else d for d in defaults)
    live = getattr(reprlib, "REPR_CONTEXT", None)
    if live is not None:
        live.current_color = plain


# ── ipykernel: stop the kernel from re-forcing colour ───────────────────
# ZMQInteractiveShell.init_environment() unconditionally sets TERM=xterm-color,
# CLICOLOR=1, FORCE_COLOR=1 and CLICOLOR_FORCE=1 in os.environ at kernel
# startup, so anything the build passes in (NO_COLOR, TERM=dumb) is overwritten
# before the first cell runs. That is right for an interactive frontend and
# wrong for a book: it is why `NO_COLOR` alone cannot fix notebook output. Wrap
# the hook so the book's policy is re-applied immediately afterwards.
_COLOUR_POLICY = {"TERM": "dumb", "NO_COLOR": "1", "TTY_COMPATIBLE": "0"}
_COLOUR_UNSET = ("FORCE_COLOR", "CLICOLOR", "CLICOLOR_FORCE")


def _apply_colour_policy():
    for key, value in _COLOUR_POLICY.items():
        os.environ[key] = value
    for key in _COLOUR_UNSET:
        os.environ.pop(key, None)


def _quiet_ipykernel(zmqshell):
    shell = getattr(zmqshell, "ZMQInteractiveShell", None)
    original = getattr(shell, "init_environment", None)
    if original is None:
        return

    def init_environment(self, _original=original):
        _original(self)
        _apply_colour_policy()

    shell.init_environment = init_environment


# ── last line of defence: no SGR escapes on a cell's stdout/stderr ──────
# A few libraries hardcode colour with no switch at all — torch.serialization,
# for one, emboldens part of its weights_only error message with a literal
# "\033[1m". Book output is plain text in both media, and an SGR sequence
# carries no information a reader can use, so strip exactly those (CSI … m)
# from the kernel's output streams. Cursor-motion and erase sequences are
# deliberately left alone: those are progress bars, and a progress bar is a
# call-site bug (pass verbose=2 / disable tqdm), not something to paper over.
def _strip_sgr_from_streams(iostream):
    import re
    sgr = re.compile(r"\x1b\[[0-9;]*m")
    stream_cls = getattr(iostream, "OutStream", None)
    original = getattr(stream_cls, "write", None)
    if original is None or getattr(original, "_d2l_sgr_stripped", False):
        return

    def write(self, string, _original=original, _sgr=sgr):
        if string and "\x1b[" in string:
            string = _sgr.sub("", string)
        return _original(self, string)

    write._d2l_sgr_stripped = True
    stream_cls.write = write


# ── rich: never use the Jupyter renderer ────────────────────────────────
# rich.console.Console._detect_color_system() returns TRUECOLOR unconditionally
# when it thinks it is in a notebook, so NO_COLOR / TERM=dumb cannot reach it,
# and the text/plain half of its mime bundle (the half the output store keeps)
# is full of escapes. Forcing the terminal renderer makes rich fall back to
# is_terminal=False → color_system=None: the same tables, in plain text.
def _quiet_rich(console):
    console._is_jupyter = lambda: False


# ── repr widths ─────────────────────────────────────────────────────────
# These match the library defaults today; setting them explicitly pins the
# book's output shape to the book rather than to whatever a future release
# happens to pick.
def _numpy_width(np):
    np.set_printoptions(linewidth=75)


def _torch_width(torch):
    torch.set_printoptions(linewidth=_WRAP_WIDTH)


def _pandas_width(pd):
    pd.set_option("display.width", _WRAP_WIDTH)


# ── compact, path-free warnings ─────────────────────────────────────────
# The default format is "<abs path>:<lineno>: <Category>: <message>\n  <src>\n",
# which puts a machine-specific 90-column path into the book. Print category and
# message only, wrapped to the verbatim budget. Warnings a chapter demonstrates
# on purpose stay visible — only their location is dropped.
_IGNORED_WARNINGS = (
    # numpy 2 vs. keras.datasets: np.dtype(..., align=0) inside cifar.py.
    dict(message=r".*align should be passed as Python or NumPy boolean.*"),
    # PIL, on the Pokemon sprites used by the GAN chapter.
    dict(message=r".*Palette images with Transparency.*"),
    # torch.profiler, re-entered once per rung in the performance chapter.
    dict(message=r".*Profiler clears events at the end of each cycle.*"),
)


def _install_warning_format():
    import textwrap
    import warnings

    def formatwarning(message, category, filename, lineno, line=None):
        name = getattr(category, "__name__", str(category))
        text = " ".join(str(message).split())
        body = f"{name}: {text}" if text else name
        return "\n".join(textwrap.wrap(
            body, width=_WRAP_WIDTH,
            break_long_words=False, break_on_hyphens=False)) + "\n"

    warnings.formatwarning = formatwarning
    # Kept explicit and short on purpose: a blanket "ignore everything" filter
    # would also hide the warnings some cells raise deliberately (the
    # unregistered-container Blocks demo in builders-guide, "input ran out of
    # data" in the NLP chapters).
    for kwargs in _IGNORED_WARNINGS:
        try:
            warnings.filterwarnings("ignore", **kwargs)
        except Exception:
            pass


def _apply():
    _apply_colour_policy()
    _on_import("ipykernel.zmqshell", _quiet_ipykernel)
    _on_import("ipykernel.iostream", _strip_sgr_from_streams)
    _on_import("flax.nnx.reprlib", _quiet_flax_nnx)
    _on_import("rich.console", _quiet_rich)
    _on_import("numpy", _numpy_width)
    _on_import("torch", _torch_width)
    _on_import("pandas", _pandas_width)
    _install_warning_format()
    sys.meta_path.insert(0, _PostImportFinder())


if not os.environ.get("D2L_NO_QUIET"):
    _apply()
