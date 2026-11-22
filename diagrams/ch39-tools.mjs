// Slide diagrams for chapter_appendix-tools-for-deep-learning (§39 Tools for Deep Learning)
import { C, TOKENS, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// Fit font-size so text width never exceeds maxW (with safety margin for bold/mono/caps)
function fitText(x, y, text, maxW, o = {}) {
  const baseFs = o.fs || 13;
  const fw = o.fw || 600;
  const mono = !!o.mono;
  const factor = mono ? 0.68 : (fw >= 700 ? 0.62 : 0.58);
  const est = text.length * factor;
  const fs = est * baseFs > maxW ? Math.max(7.5, +((maxW / est)).toFixed(1)) : baseFs;
  return tx(x, y, text, { ...o, fs, fw });
}

// Reusable card helper: guarantees text stays strictly inside box borders
function card(x, y, w, h, fill, stroke, title, lines, titleFill = C.ink, monoLines = false) {
  let o = box(x, y, w, h, fill, stroke, 8, 2);
  const padW = w - 20;
  const n = lines.length;
  const topPad = n <= 1 ? 22 : (n === 2 ? 20 : 18);
  const step = n <= 1 ? 24 : (n === 2 ? 22 : 19);
  o += fitText(x + w / 2, y + topPad, title, padW, { fs: 14, fw: 700, fill: titleFill });
  lines.forEach((ln, i) => {
    const isObj = typeof ln === 'object';
    const text = isObj ? ln.t : ln;
    const mono = isObj ? ln.mono : monoLines;
    const col = isObj && ln.fill ? ln.fill : C.ink;
    const fs = isObj && ln.fs ? ln.fs : 12.2;
    const fw = isObj && ln.fw ? ln.fw : 600;
    o += fitText(x + w / 2, y + topPad + 21 + i * step, text, padW, { mono, fs, fw, fill: col });
  });
  return o;
}

// Reusable 2x2 grid diagram builder
function grid2x2(mainTitle, c1, c2, c3, c4, footer = null) {
  const W = 560, H = footer ? 286 : 254;
  let o = fitText(280, 18, mainTitle, W - 32, { fs: 15, fw: 700, fill: C.ink });
  const ch = 96;
  o += card(20,  36, 250, ch, c1.fill, c1.st, c1.title, c1.lines, c1.dk);
  o += card(290, 36, 250, ch, c2.fill, c2.st, c2.title, c2.lines, c2.dk);
  o += card(20,  144, 250, ch, c3.fill, c3.st, c3.title, c3.lines, c3.dk);
  o += card(290, 144, 250, ch, c4.fill, c4.st, c4.title, c4.lines, c4.dk);
  if (footer) {
    o += box(20, 250, 520, 28, footer.fill, footer.st, 6, 1.8);
    o += fitText(280, 264, footer.text, 496, { mono: !!footer.mono, fs: 12.5, fw: 700, fill: footer.dk });
  }
  return svg(W, H, o);
}

// Reusable 2-column side-by-side diagram builder with footer banner
function twoCols(mainTitle, left, right, footer) {
  const W = 560, H = 280;
  let o = fitText(280, 18, mainTitle, W - 32, { fs: 15, fw: 700, fill: C.ink });
  o += card(20,  38, 250, 156, left.fill, left.st, left.title, left.lines, left.dk);
  o += card(290, 38, 250, 156, right.fill, right.st, right.title, right.lines, right.dk);
  o += box(20, 208, 520, 58, footer.fill, footer.st, 8, 1.8);
  o += fitText(280, 228, footer.line1, 496, { mono: !!footer.mono1, fs: 13, fw: 700, fill: footer.dk });
  o += fitText(280, 250, footer.line2, 496, { mono: !!footer.mono2, fs: 12.2, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// Reusable 3-column diagram builder with footer banner
function threeCols(mainTitle, c1, c2, c3, footer, arrows = false) {
  const W = 560, H = 282;
  let o = fitText(280, 18, mainTitle, W - 32, { fs: 14.8, fw: 700, fill: C.ink });
  o += card(16,  38, 168, 158, c1.fill, c1.st, c1.title, c1.lines, c1.dk);
  o += card(196, 38, 168, 158, c2.fill, c2.st, c2.title, c2.lines, c2.dk);
  o += card(376, 38, 168, 158, c3.fill, c3.st, c3.title, c3.lines, c3.dk);
  if (arrows) {
    o += arrow(186, 117, 194, 117, c1.st);
    o += arrow(366, 117, 374, 117, c2.st);
  }
  o += box(16, 210, 528, 58, footer.fill, footer.st, 8, 1.8);
  o += fitText(280, 230, footer.line1, 504, { mono: !!footer.mono1, fs: 13, fw: 700, fill: footer.dk });
  o += fitText(280, 252, footer.line2, 504, { mono: !!footer.mono2, fs: 12.2, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// Reusable 3-row vertical stack diagram builder
function threeRows(mainTitle, r1, r2, r3, arrows = true) {
  const W = 560, H = 284;
  let o = fitText(280, 18, mainTitle, W - 32, { fs: 15, fw: 700, fill: C.ink });
  o += card(20, 36,  520, 68, r1.fill, r1.st, r1.title, r1.lines, r1.dk);
  if (arrows) o += arrow(280, 106, 280, 120, r1.st);
  o += card(20, 122, 520, 68, r2.fill, r2.st, r2.title, r2.lines, r2.dk);
  if (arrows) o += arrow(280, 192, 280, 206, r2.st);
  o += card(20, 208, 520, 66, r3.fill, r3.st, r3.title, r3.lines, r3.dk);
  return svg(W, H, o);
}

export const diagrams = {
  // ── §39.0 index.md (4 diagrams) ─────────────────────────────────────────
  'ch39-tools-four-questions': () => grid2x2(
    'Four Durable Systems Questions Behind Every Tooling Choice',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Does It Fit in Memory?', lines: [
      { t: 'M_train ≈ 16N + M_act + M_ws', mono: true, fill: D.blue.dark, fw: 700 },
      'OOM stops a job before FLOP/s matters',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. What Physical Roof Binds?', lines: [
      { t: 'Prefill/Train: P  ·  Decode: β', mono: true, fill: D.orange.dark, fw: 700 },
      'Batch-1 decode reads every weight/token',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Cost per Completed Result?', lines: [
      { t: 'Invoice + Storage + Egress + Eng', mono: true, fill: D.green.dark, fw: 700 },
      '1 extra debug round outweighs $/hr gap',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '4. Can Others Reproduce It?', lines: [
      { t: 'Fresh Kernel + uv Lock + 40-c SHA', mono: true, fill: D.purple.dark, fw: 700 },
      'Eliminate hidden state & floating refs',
    ]},
    { fill: C.lgray, st: C.gray, dk: C.ink, text: 'Products & prices change monthly; these four physical & engineering laws stay fixed.' }
  ),

  'ch39-tools-sizing-equations': () => grid2x2(
    'Four First-Principles Equations for Back-of-the-Envelope Sizing',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Peak Training Memory (§§39.4, 39.6)', lines: [
      { t: 'M_train ≈ 16N + M_act + M_ws', mono: true, fill: D.blue.dark, fw: 700 },
      '2B w + 2B g + 4B master + 8B Adam',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Decode Bandwidth Ceiling (§§39.4, 39.7)', lines: [
      { t: 'tok/s ≲ β / (B_w + b·n·B_kv)', mono: true, fill: D.orange.dark, fw: 700 },
      'Active weight bytes + batched KV cache',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Training FLOP Budget (§§39.2, 39.6)', lines: [
      { t: 'C ≈ 6 · N · D  FLOPs', mono: true, fill: D.green.dark, fw: 700 },
      '2ND forward + 4ND backward pass',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: "Young's Checkpoint Law (§§39.2, 39.3)", lines: [
      { t: 'τ* = √(2 · δ · M)', mono: true, fill: D.purple.dark, fw: 700 },
      'Write cost δ vs. mean failure interval M',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, text: 'Always size memory, throughput, FLOPs, and checkpoint cadence before launching!' }
  ),

  'ch39-tools-lifecycle-roadmap': () => threeRows(
    'Chapter 39 Roadmap: From a Single Notebook to Fleet Serving',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Tier 1: Reproducible Interactive Runtimes (§§39.1–39.2)', lines: [
      '§39.1 Local notebooks, kernel state, uv locks, SSH  ·  §39.2 Colab & Kaggle ephemeral VMs',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Tier 2: Compute Economics, Silicon & Supply Chain (§§39.3–39.5)', lines: [
      '§39.3 Cloud tiers, spot & egress  ·  §39.4 Discrete vs. unified RAM  ·  §39.5 Hubs, Elo & safetensors',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Tier 3: Cluster Training, Serving & D2L Build Pipeline (§§39.6–39.9)', lines: [
      '§39.6 DDP/FSDP2/3D parallel  ·  §39.7 vLLM/SGLang & PagedAttention  ·  §§39.8–39.9 Build & d2l utils',
    ]}
  ),

  'ch39-tools-worked-benchmarks': () => grid2x2(
    'Executable Sizing Models & Empirical Snapshots Across Chapter 39',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '§§39.1–39.2: Kernel & Portable Setup', lines: [
      'uv pip sync + sys.executable check',
      'Environment fingerprint & path adapter',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '§§39.3–39.4: Cloud & Hardware Sizing', lines: [
      '4090 / A100 / H100 completed-result cost',
      '7B full (86 GiB) vs QLoRA (4 GiB) & decode',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '§§39.5–39.6: Arena Elo & 8-GPU FSDP', lines: [
      'Arena SE ≈ 347/√v & frozen Artifact SHA',
      '8-GPU 7B: DDP (95 GiB) vs FSDP (22 GiB)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '§§39.7–39.9: Serving & Single-Source Build', lines: [
      'Continuous batching & PagedAttention math',
      'Prefix fingerprinting & JAX RL bucketing',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, text: 'Every section pairs durable systems theory with runnable Python verification.' }
  ),

  // ── §39.1 interactive-development.md (9 diagrams) ──────────────────────
  'ch39-tools-script-vs-notebook': () => twoCols(
    'Python Script vs. Interactive Jupyter Kernel Execution',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Batch Python Script (python train.py)', lines: [
      'Starts a fresh interpreter every run',
      'Executes strictly top-to-bottom',
      'No state survives between runs',
      'Re-runs slow data loading every time',
      { t: 'Always reproducible by construction', fill: D.blue.dark, fw: 700 },
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Interactive Notebook Kernel (.ipynb)', lines: [
      'Persistent Python process across edits',
      'Keeps datasets & tensors resident in RAM',
      'Cells run in arbitrary user click order',
      'Deleted cells leave variables behind!',
      { t: 'Needs Restart & Run All verification', fill: D.orange.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Discipline: Explore interactively cell-by-cell, verify top-to-bottom from a blank kernel.',
      line2: 'The D2L build executes every notebook in a fresh kernel before publishing outputs.' }
  ),

  'ch39-tools-hidden-state-types': () => threeCols(
    'Three Mechanisms of Hidden Kernel State (Pimentel et al., 2019)',
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '1. Out-of-Order Run', lines: [
      { t: 'In [5] above In [4]', mono: true, fill: D.orange.dark, fw: 700 },
      'Lower cell sets lr=1e-4,',
      'then upper train cell runs',
      { t: 'Silent value drift on', fill: D.red.dark, fw: 700 },
      { t: 'fresh top-down run!', fill: D.red.dark, fw: 700 },
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: '2. In-Place Mutation', lines: [
      { t: 'pixels = pixels / 255', mono: true, fill: D.red.dark, fw: 700 },
      'Running the same cell',
      'twice scales by 1/255²!',
      'Fresh run scales by',
      '1/255 → different loss',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '3. Edited / Deleted Cell', lines: [
      { t: 'Ghost RAM bindings', mono: true, fill: D.purple.dark, fw: 700 },
      'Deleting helper cell or',
      'editing code without run',
      'Crashes with NameError',
      'on fresh kernel restart',
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark,
      line1: 'Why Silent Drift (1 & 2) Is Worse Than a NameError (3):',
      line2: 'A NameError halts immediately; an overwritten hyperparameter logs plausible but false metrics!' }
  ),

  'ch39-tools-restart-run-all': () => twoCols(
    'What "Restart Kernel & Run All" Catches vs. What Still Escapes',
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Caught by Local Restart & Run All', lines: [
      '1. Upward use-before-define NameErrors',
      '2. Deleted imports or helper functions',
      '3. Repeated in-place cell mutations',
      '4. Edited cells whose output was not',
      '   re-executed after the source change',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Escapes Local Run (Needs Clean VM)', lines: [
      '1. Local cached file (features.npy) not',
      '   in Git whose compute branch has a bug',
      '2. Unrecorded pip install in old session',
      '3. Hard-coded /home/alice/data.csv path',
      '4. Unseeded RNG contradicting prose',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Full Reproducibility Test: Fresh Clone + Fresh uv Lock Env + Top-to-Bottom Replay',
      line2: 'Keep function definitions separate from experiment calls and seed generators explicitly.' }
  ),

  'ch39-tools-uv-lock-flow': () => threeCols(
    'Deterministic Local Setup & Kernel Identity Verification',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Create & Sync Lock', lines: [
      { t: 'uv venv --python 3.12', mono: true, fill: D.blue.dark, fw: 700 },
      { t: 'uv pip sync pylock...', mono: true, fill: D.blue.dark, fw: 700 },
      'Matches lock exactly &',
      'removes stray packages',
      'from .venv',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '2. Launch Frozen Lab', lines: [
      { t: 'uv run --no-sync \\', mono: true, fill: D.purple.dark, fw: 700 },
      { t: '  jupyter lab', mono: true, fill: D.purple.dark, fw: 700 },
      '--no-sync prevents uv',
      'from re-resolving or',
      'altering locked wheels',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Audit Active Kernel', lines: [
      { t: 'sys.executable', mono: true, fill: D.green.dark, fw: 700 },
      { t: 'os.getcwd()', mono: true, fill: D.green.dark, fw: 700 },
      'Confirms notebook uses',
      '.venv kernel (not system',
      'Python) & right folder',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Server Process ≠ Kernel Process: One Jupyter server lists many kernels!',
      line2: 'Always verify sys.executable in cell 1 and sanitize ~ before committing output.' },
    true
  ),

  'ch39-tools-orphaned-kernels': () => twoCols(
    'The Orphaned GPU Kernel Trap in JupyterLab',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Closing the Browser Tab ≠ Stopping', lines: [
      'User closes experiment_v1.ipynb tab',
      'Kernel process stays alive on server!',
      { t: 'Holds 18 GiB of 24 GiB GPU VRAM', mono: true, fill: D.red.dark, fw: 700 },
      'Next notebook crashes immediately',
      'with CUDA OutOfMemoryError!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Explicit Kernel Lifecycle Discipline', lines: [
      '1. Interrupt (I, I) hung loops to debug',
      '2. Restart (0, 0) to clear GPU tensors',
      '3. Open JupyterLab "Running" sidebar',
      { t: '4. Click "Shut Down" on idle kernels', fill: D.green.dark, fw: 700 },
      '5. Verify with nvidia-smi in terminal',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Jupyter decouples browser tabs from kernel OS processes so long runs survive reloads.',
      line2: 'Always audit the Running Terminals and Kernels panel before starting a new GPU notebook.' }
  ),

  'ch39-tools-notebook-to-module': () => twoCols(
    'Why Moving Helpers to .py Modules Fixes Multiprocessing',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Function Defined in Notebook Cell', lines: [
      { t: '__main__.square(i)', mono: true, fill: D.red.dark, fw: 700 },
      'pickle serializes functions by module name',
      'Works under Linux legacy fork, FAILS',
      'under spawn (macOS/Win) & forkserver',
      { t: 'AttributeError: no square in __main__!', mono: true, fs: 11.5, fill: D.red.dark },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Function Imported from helpers.py', lines: [
      { t: 'from helpers import square', mono: true, fill: D.green.dark, fw: 700 },
      'Worker process imports helpers.py cleanly',
      'Works under fork, spawn, and forkserver',
      '(Python 3.14+ Linux default!)',
      'Enables pytest, git diff & type checks',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Hybrid Architecture: Keep reusable data transforms & models in .py modules;',
      line2: 'use the notebook for interactive narrative, plots, and experiment records.' }
  ),

  'ch39-tools-postmortem-debug': () => threeCols(
    'Post-Mortem Inspection in Live Kernel Memory with %debug',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: '1. Exception Raised', lines: [
      { t: 'RuntimeError / NaN', mono: true, fill: D.red.dark, fw: 700 },
      'Cell execution stops,',
      'but kernel process and',
      'sys.last_traceback stay',
      'alive in RAM/VRAM!',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '2. Run %debug Next', lines: [
      { t: '%debug  (opens ipdb)', mono: true, fill: D.blue.dark, fw: 700 },
      'Jumps directly into the',
      'failing stack frame:',
      { t: 'p x.shape, x.dtype', mono: true, fs: 11.8 },
      { t: 'p torch.isnan(x).sum()', mono: true, fs: 11.8 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Fix Without Reload', lines: [
      'No need to re-run',
      '10-minute dataset prep!',
      'Never wrap code in',
      'broad try/except that',
      'swallows the traceback',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Anti-Pattern:  try: train()  except Exception: print("failed")',
      line2: 'Swallowing exceptions destroys the frame, line number, and tensor state needed for %debug.' },
    true
  ),

  'ch39-tools-ssh-port-collisions': () => threeCols(
    'SSH Port Forwarding & Two Multi-User Port Collision Traps',
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Safe SSH Loopback', lines: [
      { t: '--ip 127.0.0.1', mono: true, fill: D.green.dark, fw: 700 },
      { t: 'ssh -N -L P:127.0.0.1:P', mono: true, fs: 11.5 },
      'Never bind 0.0.0.0!',
      'SSH encrypts traffic +',
      'token guards localhost',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Trap 1: Remote 8888 Busy', lines: [
      'Alice holds remote 8888;',
      'your server picks 8889',
      'Tunnel to 8888 reaches',
      { t: "Alice's Jupyter server!", fill: D.orange.dark, fw: 700 },
      'Fix: --port-retries=0',
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Trap 2: Local 8888 Busy', lines: [
      'Laptop already runs',
      'local Jupyter on 8888;',
      'ssh warns & continues',
      { t: 'Browser runs on LAPTOP!', fill: D.red.dark, fw: 700 },
      'Fix: ExitOnForwardFailure',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Safe Command:  ssh -N -o ExitOnForwardFailure=yes -L 8899:127.0.0.1:8899 myserver',
      line2: 'Also check platform.node() (hostname) alongside sys.executable to distinguish laptop vs. server.' }
  ),

  'ch39-tools-notebook-checklist': () => grid2x2(
    'Pre-Commit Reproducibility & Privacy Gate for Shared Notebooks',
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '1. Clean Top-Down Replay', lines: [
      'Run "Restart Kernel and Run All Cells"',
      'Confirm counters run In [1] .. In [n]',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '2. Portable Paths & Downloads', lines: [
      'No /Users/alice paths or hidden local files',
      'Sanitize ~ and basename in output logs',
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: '3. Scrub Secrets & Trim Noise', lines: [
      'Zero API keys or tokens in cell outputs',
      'Delete 1,000-line download progress bars',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '4. Record Environment & Hardware', lines: [
      'Pin uv lock file / framework versions',
      'Log GPU model, dtype & shapes for timings',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, text: 'A public notebook is an executable publication: its saved outputs are part of what you publish.' }
  ),

  // ── §39.2 hosted-notebooks.md (9 diagrams) ──────────────────────────────
  'ch39-tools-ephemeral-vm': () => twoCols(
    'Hosted Notebook Runtimes: What Vanishes vs. What Survives',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Ephemeral Session VM (Wiped on Exit)', lines: [
      'Destroyed after idle timeout or 12h limit',
      'Wipes ad-hoc !pip install packages',
      'Wipes datasets in /content or /tmp',
      'Wipes local checkpoints & kernel RAM',
      { t: 'Never trust local VM disk overnight!', fill: D.red.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Durable External State (Survives)', lines: [
      '1. Idempotent setup cell in the notebook',
      '2. Mounted Google Drive (/content/drive)',
      '3. Kaggle /kaggle/input & saved versions',
      '4. External object storage / Hugging Face',
      { t: 'Reconstructs environment in < 60s!', fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Core Rule: Treat every hosted VM as replaceable cattle, never as a pet machine.',
      line2: 'If your first cell rebuilds packages and your loop checkpoints to durable storage, resets cost nothing.' }
  ),

  'ch39-tools-badge-security': () => threeCols(
    'Three Supply-Chain Risks in Untrusted "Open in Colab" Badges',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: '1. Unpinned git+https', lines: [
      { t: 'pip install git+https...', mono: true, fs: 11.5, fill: D.red.dark, fw: 700 },
      'Runs arbitrary setup.py',
      'code from floating main',
      'branch!',
      { t: 'Fix: Pin 40-char SHA', fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. Pipe to Shell', lines: [
      { t: 'curl https://... | bash', mono: true, fs: 11.5, fill: D.orange.dark, fw: 700 },
      'Executes remote shell',
      'commands invisible in',
      'the notebook page!',
      { t: 'Fix: Inline & inspect', fill: D.green.dark, fw: 700 },
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '3. Drive / Local Mount', lines: [
      { t: 'drive.mount("/content..")', mono: true, fs: 11.2, fill: D.purple.dark, fw: 700 },
      'Grants notebook full',
      'read/write access to all',
      'personal Drive files (or',
      'laptop on Local Runtime!)',
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark,
      line1: 'Always inspect the repository origin and setup cell before pressing Run!',
      line2: 'Never attach an untrusted notebook to a Local Runtime (shifts execution to your own OS account).' }
  ),

  'ch39-tools-colab-tiers': () => twoCols(
    'Google Colab Resource Model: Free Tier vs. Paid Compute Units',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Colab Free Tier (Nov 2026)', lines: [
      'Commonly assigns NVIDIA T4 (16 GB VRAM)',
      'Quotas & GPU availability fluctuate daily',
      'Max 12-hour session; aggressive idle cut',
      'Opens GitHub notebooks as read-only links',
      { t: 'Must "Save a copy in Drive" to keep edits!', fill: D.red.dark, fw: 700 },
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: 'Compute Units (Pay-As-You-Go / Pro+)', lines: [
      'Purchases priority access to L4 & A100',
      'Higher memory VMs & background runs',
      'Still best-effort (not a dedicated lease)',
      'Colab CLI (June 2026): colab new/exec/stop',
      { t: 'Pin dated runtime images for courses', fill: D.purple.dark, fw: 700 },
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'The UI menu shows what accelerator you requested, not what you received.',
      line2: 'Always check sys.modules, torch.cuda.get_device_properties(0), or jax.devices() in code.' }
  ),

  'ch39-tools-colab-version-cli': () => twoCols(
    'Managing Colab Image Drift & Driving Runtimes From the CLI',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Dated Runtime Version Pinning', lines: [
      'Colab updates default base images often',
      'Runtime → Change runtime type → Version',
      'Dated images stay available for 1 year',
      'Connecting to a past image takes 2–3 min',
      { t: 'Ideal for course cohorts & benchmarks', fill: D.blue.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Colab CLI Workflow (Linux / macOS)', lines: [
      { t: '1. colab new   (allocate cloud VM)', mono: true, fill: D.green.dark, fw: 700 },
      { t: '2. colab exec  (run script/notebook)', mono: true, fill: D.green.dark, fw: 700 },
      { t: '3. colab stop  (release & stop quota)', mono: true, fill: D.green.dark, fw: 700 },
      'Lets local terminals & coding agents run',
      'GPU tests without leaving the editor',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Recommendation: Use latest runtime + pinned library versions for active development;',
      line2: 'pin a dated runtime version when distributing fixed teaching material to hundreds of readers.' }
  ),

  'ch39-tools-kaggle-fs-zones': () => threeCols(
    'Kaggle File System Zones & Immutable Save-and-Run Versioning',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. /kaggle/input/', lines: [
      { t: 'READ-ONLY & VERSIONED', mono: true, fs: 11.5, fill: D.blue.dark, fw: 700 },
      'Attached datasets,',
      'pretrained weights &',
      'prior notebook outputs',
      '(0 download time!)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '2. /kaggle/working/', lines: [
      { t: 'WRITABLE (20 GB Cap)', mono: true, fs: 11.5, fill: D.green.dark, fw: 700 },
      'Checkpoints & results',
      'written here persist into',
      { t: 'Save & Run All', fill: D.green.dark, fw: 700 },
      'immutable versions!',
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: '3. /tmp & Other Paths', lines: [
      { t: 'EPHEMERAL SCRATCH', mono: true, fs: 11.5, fill: D.red.dark, fw: 700 },
      'Fast local space during',
      'the session, but wiped',
      'completely as soon as',
      'the session ends',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Save & Run All = Built-In "Restart & Run All" + Immutable Artifact Snapshot',
      line2: 'Attaching data & wheel packages via /kaggle/input/ allows 100% offline competition runs.' }
  ),

  'ch39-tools-bert-t4-budget': () => twoCols(
    'Sizing Workloads Against a Free 30-Hour Weekly T4 GPU Quota',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'BERT-Base Pretraining (N = 110M)', lines: [
      { t: 'D ≈ 4.28 × 10¹⁰ tokens (1M steps)', mono: true, fill: C.ink },
      { t: 'C ≈ 6ND ≈ 2.8 × 10¹⁹ FLOPs', mono: true, fill: D.red.dark, fw: 700 },
      'T4 at 30% of 65 TF/s = 19.5 TFLOP/s:',
      { t: 'Requires ~400 T4 GPU-Hours!', mono: true, fill: D.red.dark, fw: 700 },
      '= 33+ 12h sessions (>13 weeks quota)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'What 1 Twelve-Hour T4 Session Fits', lines: [
      { t: 'Budget ≈ 8.4 × 10¹⁷ FLOPs', mono: true, fill: D.green.dark, fw: 700 },
      '1. All D2L chapter notebooks & exercises',
      '2. QLoRA fine-tuning of 7B–8B models',
      '3. Chinchilla-optimal (D = 20N) pretrain',
      { t: '   of an N ≈ 84M model on 1.7B tokens!', fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Always compute C ≈ 6ND and divide by sustained TFLOP/s before starting a multi-day run!',
      line2: 'When a job exceeds 1–2 sessions, rent dedicated cloud capacity (§39.3) instead of chaining free VMs.' }
  ),

  'ch39-tools-colab-vs-kaggle': () => twoCols(
    'Choosing Between Google Colab and Kaggle Notebooks (Nov 2026)',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Choose Google Colab When…', lines: [
      '• Opening a D2L chapter directly from',
      '  GitHub with zero account import delay',
      '• Saving notebooks & data in Google Drive',
      '• Bursting to L4/A100 via Compute Units',
      '• Driving quick GPU runs via colab CLI',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Choose Kaggle Notebooks When…', lines: [
      '• You need a predictable, published',
      '  30 h/week GPU quota (P100 or 2× T4)',
      '• Mounting versioned datasets at /kaggle/input',
      '• Capturing immutable Save & Run All logs',
      '• Participating in offline competitions',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Portable notebooks run on BOTH platforms unmodified via a 6-line pathlib adapter',
      line2: 'and an idempotent setup cell that inspects package metadata before installing.' }
  ),

  'ch39-tools-young-checkpoint': () => twoCols(
    "Surviving Session Disconnects: Young's Checkpoint Interval τ*",
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: "Young's Formula (1974): τ* = √(2·δ·M)", lines: [
      { t: 'δ = checkpoint write duration', mono: true, fill: C.ink },
      { t: 'M = mean time between disconnects', mono: true, fill: C.ink },
      'Balances time spent writing checkpoints',
      'against expected compute lost on crash',
      { t: 'δ = 30s, M = 3h → τ* ≈ 13.4 minutes', mono: true, fill: D.blue.dark, fw: 700 },
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Known 12h Cap + Random Disconnects', lines: [
      '1. Checkpoint every τ* = √(2δM) for',
      '   random idle/preemption disconnects',
      '2. Force a deterministic final checkpoint',
      '   just before the known 12-hour wall!',
      { t: '3. Verify target path is durable (Drive)!', fill: D.red.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Why τ* Is Robust: Total overhead curve E[T]/τ is flat near the minimum τ*,',
      line2: 'so checkpointing every 10–15 minutes wastes < 4% of wall time while bounding lost work!' }
  ),

  'ch39-tools-portable-summary': () => grid2x2(
    'Four Engineering Pillars of a Portable Hosted Notebook',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Idempotent Setup Cell', lines: [
      'Check importlib.metadata.version first',
      'Pin exact versions; restart if already in RAM',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '2. Environment Fingerprint', lines: [
      'Log Python, NumPy, framework & GPU specs',
      'Record COLAB_* / KAGGLE_* image vars',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '3. Durable Path Adapter', lines: [
      'Auto-detect /kaggle/working & /content/drive',
      'Refuse to save checkpoints to ephemeral /tmp',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '4. Secret Store & Audit', lines: [
      'Read tokens from provider Secret Manager',
      'Never print credentials in published outputs',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, text: 'A notebook built on these four rules runs on Colab, Kaggle, and local laptops alike.' }
  ),

  // ── §39.3 cloud-instances.md (9 diagrams) ───────────────────────────────
  'ch39-tools-three-cloud-walls': () => threeCols(
    'Three Walls That Push a Project From Free Tiers to Cloud Rentals',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Duration Wall', lines: [
      { t: '> 12h Session / 30h Wk', mono: true, fill: D.blue.dark, fw: 700 },
      'Multi-day pretraining',
      'or large dataset prep',
      'outlasts ephemeral',
      'browser VM quotas',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. VRAM Capacity Wall', lines: [
      { t: '80–180 GB HBM Needed', mono: true, fill: D.orange.dark, fw: 700 },
      'Full 7B+ fine-tuning or',
      'long-context batch eval',
      'exceeds 16–24 GB cards',
      '(A100 / H100 / B200)',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '3. Multi-GPU Link Wall', lines: [
      { t: '4–8× GPUs + NVLink', mono: true, fill: D.purple.dark, fw: 700 },
      'FSDP & tensor parallel',
      'require 450–900 GB/s',
      'NVLink nodes rather',
      'than single PCIe cards',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Note: Every single-GPU D2L notebook peaks < 13 GiB and runs on a $0.40/hr 24 GB rental.',
      line2: 'Rent larger instances only when one of these three physical walls actually binds.' }
  ),

  'ch39-tools-trust-boundaries': () => twoCols(
    'Cloud Trust Boundaries: Community Marketplaces vs. Vetted Clouds',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Community Marketplace Host', lines: [
      'Physical server operated by an unknown 3rd party',
      'Host root can inspect container RAM & GPU VRAM',
      'Disk encryption DOES NOT hide live RAM keys!',
      { t: 'Risk: Leaked S3 keys, private weights/code', fill: D.red.dark, fw: 700 },
      'Safe only for public coursework & open data',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Vetted Datacenter & Hyperscalers', lines: [
      'SOC2 / ISO physical & hypervisor isolation',
      'Scoped, short-lived IAM credentials & KMS',
      'VPC private networking & audit logging',
      { t: 'Regulated data (HIPAA): signed BAA first!', fill: D.green.dark, fw: 700 },
      'Required for proprietary models & customer PII',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Security Rule: Never place long-lived cloud root credentials on any rented GPU instance.',
      line2: 'Use read-only or bucket-scoped tokens that expire automatically when the job finishes.' }
  ),

  'ch39-tools-price-spread': () => threeCols(
    '3× to 5× On-Demand Price Spread for Identical Silicon (Nov 2026)',
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '1. Marketplaces', lines: [
      { t: 'Vast.ai / RunPod', mono: true, fill: D.green.dark, fw: 700 },
      'RTX 4090: $0.30–0.89/h',
      { t: 'H100 80G: $1.80–3.99/h', mono: true, fill: D.green.dark, fw: 700 },
      'Lowest price; variable',
      'host disk & network I/O',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '2. GPU Specialists', lines: [
      { t: 'Lambda / Nebius / VP', mono: true, fill: D.blue.dark, fw: 700 },
      'A100 / PRO 6000: ~$1.90',
      { t: 'H100 80G: $1.99–4.50/h', mono: true, fill: D.blue.dark, fw: 700 },
      'Self-serve ML images +',
      'fast NVLink clusters',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '3. Hyperscalers', lines: [
      { t: 'AWS / Azure / GCP', mono: true, fill: D.purple.dark, fw: 700 },
      'L4 / T4: $0.53–0.80/h',
      { t: 'H100 80G: $6.88–11.00/h', mono: true, fill: D.purple.dark, fw: 700 },
      'Full cloud IAM/VPC/S3;',
      'GCP TPU v5e at $1.20/h',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Two Caveats: (1) Verify whether an 8×H100 listing uses NVLink (450 GB/s) or PCIe (64 GB/s).',
      line2: '(2) Newer B200 ($7–14/hr) only beats H100 when your job uses its 180 GB HBM or FP8/FP4 peak.' }
  ),

  'ch39-tools-nvlink-vs-pcie-cloud': () => twoCols(
    '8× H100 Node at Equal $/hr: NVLink (450 GB/s) vs. PCIe 5.0 (64 GB/s)',
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '8× H100 SXM with NVLink (450 GB/s)', lines: [
      { t: 'Sustained F ≈ 395.6 TFLOP/s (40%)', mono: true, fill: C.ink },
      { t: 'b_balance = F·β_link / 3 ≈ 293 tok/GPU', mono: true, fill: D.green.dark, fw: 700 },
      'At b = 4,096 tokens/GPU (N = 1.3B):',
      'Compute = 78.9 ms, Comm = 5.8 ms',
      { t: 'Comm overhead is only 7.3%!', fill: D.green.dark, fw: 700 },
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: '8× H100 PCIe 5.0 (64 GB/s)', lines: [
      { t: 'Same GPU compute F ≈ 395.6 TFLOP/s', mono: true, fill: C.ink },
      { t: 'b_balance = F·β_link / 3 ≈ 2,060 tok/GPU', mono: true, fill: D.red.dark, fw: 700 },
      'At b = 4,096 tokens/GPU (N = 1.3B):',
      'Compute = 78.9 ms, Comm = 40.6 ms',
      { t: 'Comm overhead is 51.5% (30% slower)!', fill: D.red.dark, fw: 700 },
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Notice that parameter count N cancels out of t_comm / t_compute = (2N·β_link) / (6Nb / F)!',
      line2: 'A PCIe node must be ≥ 30% cheaper per hour (or accumulate N_acc microbatches) to match NVLink.' }
  ),

  'ch39-tools-completed-result-cost': () => twoCols(
    'Anatomy of Total Cost per Completed Result vs. Sticker Price',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Direct Cloud Invoice Terms', lines: [
      { t: '1. Execution Compute: T_wall · p_gpu', mono: true, fill: D.blue.dark, fw: 700 },
      { t: '2. Setup & Debug Compute: T_setup · p_gpu', mono: true, fill: C.ink },
      { t: '3. Attached Disk Storage: C_storage', mono: true, fill: C.ink },
      '   (bills 24/7 even while VM is stopped!)',
      { t: '4. Outbound Data Egress: C_egress', mono: true, fill: D.red.dark, fw: 700 },
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: 'Human Engineering Time Terms', lines: [
      { t: '5. Hands-On Setup/Debug: T_setup · p_eng', mono: true, fill: D.purple.dark, fw: 700 },
      '   ($60/hr × 0.5 hr = $30 per debug round)',
      { t: '6. Run Monitoring: f · T_wall · p_eng', mono: true, fill: D.purple.dark, fw: 700 },
      '   Context switching & waiting on results',
      '   Fast GPUs turn overnight runs into 2h!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Total Cost = (T_wall + T_setup)·p_gpu + C_storage + C_egress + (T_setup + f·T_wall)·p_eng',
      line2: 'Optimize for the total cost of a verified, reproducible answer—never hourly GPU rate alone.' }
  ),

  'ch39-tools-breakeven-attention': () => twoCols(
    'Why a Pricier H100 ($2.50/h) Often Beats a Cheap 4090 ($0.40/h)',
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'RTX 4090 ($0.40/h, 1.0× Speed, 8.0h)', lines: [
      { t: 'Cloud Invoice: $5.40  (Cheapest bill)', mono: true, fill: D.orange.dark, fw: 700 },
      '+ 0.5h setup ($30) = $35.40 total',
      'If 1 extra 30-min debug round occurs:',
      { t: 'Total jumps to > $65.00!', mono: true, fill: D.red.dark, fw: 700 },
      'Or if engineer spends > 7 min checking!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'H100 80GB ($2.50/h, 3.0× Speed, 2.7h)', lines: [
      { t: 'Cloud Invoice: $9.92  (+$4.52 gap)', mono: true, fill: D.green.dark, fw: 700 },
      '+ 0.5h setup ($30) = $39.92 total',
      'Saves 5.33 hours of wall-clock waiting!',
      { t: 'Breaks even at f = 1.4% attention', mono: true, fill: D.green.dark, fw: 700 },
      '(just 6.8 minutes of engineer time!)',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Invoice-Only Break-Even requires H100 relative_speed = 8.0 / 0.86 ≈ 9.3× (unrealistic);',
      line2: 'Engineer-Inclusive Break-Even requires only 6.8 minutes of saved human waiting time!' }
  ),

  'ch39-tools-spot-blowup': () => twoCols(
    'Spot Capacity Economics: Exponential Blowup vs. Checkpointed Gain',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Uncheckpointed Spot Run (Start Over)', lines: [
      { t: 'E[T_wall] = (e^(T/M) − 1)·(M + R)', mono: true, fill: D.red.dark, fw: 700 },
      '8h job, mean interruption M = 4h,',
      'restart R = 0.25h, half price q = 0.5:',
      { t: 'E[T_wall] = (e² − 1)·4.25 = 27.1 hours!', mono: true, fill: D.red.dark, fw: 700 },
      'Expected cost is 1.69× On-Demand!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Checkpointed Spot Run (τ* = √(2δM))', lines: [
      { t: 'Loses at most τ* ≈ 33 min per eviction', mono: true, fill: D.green.dark, fw: 700 },
      'Each segment (τ*+δ)/M ≪ 1 stays linear!',
      'At δ = 2 min, R = 15 min, M = 4h:',
      { t: 'Spot is cheaper for any q < 0.80!', mono: true, fill: D.green.dark, fw: 700 },
      'At q = 0.50, saves ~38% even with evictions',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Warning Windows: AWS = 2 min  ·  GCP = 30 sec  ·  Vast.ai Interruptible = 0 sec (instant kill)!',
      line2: 'Never rely on shutdown signals alone; write periodic atomic checkpoints to durable storage.' }
  ),

  'ch39-tools-egress-shock': () => twoCols(
    'The Hidden Cloud Bill: Outbound Data Egress vs. GPU Compute',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Daily 140 GB Checkpoint Download', lines: [
      'Hyperscaler internet egress: ~$0.09 / GB',
      '140 GB/day × 30 days = 4,200 GB / month',
      { t: 'Monthly Egress Bill ≈ $370.00!', mono: true, fs: 13.5, fill: D.red.dark, fw: 700 },
      'Exceeds the cost of running an RTX 4090',
      '24/7 for the entire month ($320)!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Three Rules to Eliminate Egress Taxes', lines: [
      '1. Keep compute VMs and object storage',
      '   in the EXACT SAME provider & region',
      '2. Evaluate zero-egress storage/clouds',
      '   (Cloudflare R2, RunPod) for artifacts',
      '3. Delete unattached block volumes & IPs!',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Set a billing alert on Day 1 and cap account GPU quotas to bound leaked-key risk.',
      line2: 'When an experiment finishes, tear down the VM, scratch disks, snapshots, and stale buckets.' }
  ),

  'ch39-tools-smoke-test-gates': () => threeCols(
    'The 60-Second Boot Smoke Test Before Any Multi-Hour Cloud Run',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Gates 1–2: Hardware & Disk', lines: [
      { t: '1. Accelerator visible', mono: true, fill: D.blue.dark, fw: 700 },
      '   nvidia-smi & CUDA ok',
      { t: '2. Disk space & perms', mono: true, fill: D.blue.dark, fw: 700 },
      '   Free bytes ≥ budget &',
      '   ckpt dir is writable!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Gates 3–4: Step & Restore', lines: [
      { t: '3. 1 Target Train Step', mono: true, fill: D.green.dark, fw: 700 },
      '   Catches OOM & cuDNN',
      { t: '4. Checkpoint Round-Trip', mono: true, fill: D.green.dark, fw: 700 },
      '   save → load reproduces',
      '   weights & Adam bitwise',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Gate 5: Input Throughput', lines: [
      { t: '5. Iterator vs. GPU Rate', mono: true, fill: D.orange.dark, fw: 700 },
      '   Measure data batch/s',
      '   separately from GPU',
      '   step/s (catches slow',
      '   network storage!)',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Without Gate 2, a read-only disk crashes at Hour 1 during the first checkpoint save.',
      line2: 'Without Gate 5, a slow network disk never crashes—it silently starves the GPU at 20% utilization!' },
    true
  ),

  // ── §39.4 hardware.md (8 diagrams) ──────────────────────────────────────
  'ch39-tools-three-hw-axes': () => threeCols(
    'Three Physical Quantities Govern Every Deep Learning Purchase',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Capacity (GB / GiB)', lines: [
      { t: 'Decides WHAT RUNS', mono: true, fill: D.blue.dark, fw: 700 },
      'Holds 16N training state',
      '+ activations, or active/total',
      'inference weights + KV',
      'If > RAM, speed = 0!',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. Bandwidth (GB/s)', lines: [
      { t: 'Decides DECODE SPEED', mono: true, fill: D.orange.dark, fw: 700 },
      'Each generated token',
      'streams all active weights',
      'and KV cache once:',
      { t: 'tok/s ≲ β / B_active', mono: true, fill: D.orange.dark },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Compute (TFLOP/s)', lines: [
      { t: 'Decides TRAIN & PREFILL', mono: true, fill: D.green.dark, fw: 700 },
      'Parallel token batches',
      'reuse weights in SRAM,',
      'saturating Tensor Cores',
      '(BF16 / FP8 / FP4)',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Hardware Selection Rule: Identify which of the three quantities binds your workload,',
      line2: 'then buy the machine family that maximizes that binding constraint per dollar.' }
  ),

  'ch39-tools-decode-intensity': () => twoCols(
    'Why Batch-1 Autoregressive Decoding Is Purely Bandwidth-Bound',
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Bytes Moved per Token (8B @ 4-bit)', lines: [
      { t: '4-bit Weights: 8B × 0.5 B = 4.0 GB', mono: true, fill: D.orange.dark, fw: 700 },
      { t: 'BF16 KV Cache (~4K ctx):    0.5 GB', mono: true, fill: C.ink },
      { t: 'Total B_active = 4.5 GB / token', mono: true, fill: D.orange.dark, fw: 700 },
      'At β = 900 GB/s (RTX 5070 Ti / 3090):',
      'Max decode rate ≲ 900 / 4.5 = 200 tok/s',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Arithmetic Performed per Token', lines: [
      { t: 'FLOPs = 2 × 8B = 16 GFLOPs / token', mono: true, fill: D.blue.dark, fw: 700 },
      'Arithmetic Intensity:',
      { t: 'I = 16 GFLOPs / 4.5 GB ≈ 3.5 FLOP/B', mono: true, fill: D.red.dark, fw: 700 },
      '50–80× below GPU ridge point (150–295)!',
      'Tensor Cores sit > 95% idle waiting on RAM',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Consequence: Two machines with ~260 GB/s bandwidth (DGX Spark & Strix Halo)',
      line2: 'decode at nearly the same ~50–60 tok/s rate despite a 2× price and huge TFLOP/s gap!' }
  ),

  'ch39-tools-kv-context-growth': () => threeCols(
    'Llama 3.1 8B (4-Bit Weights = 4.0 GB): KV Cache vs. Context Length',
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '4K Context (Short)', lines: [
      { t: 'KV Cache = 0.5 GB', mono: true, fill: D.green.dark, fw: 700 },
      '(128 KiB/tok × 4,096)',
      'Weights: 4.0 GB (89%)',
      'Total: 4.5 GB / token',
      '4-bit gives 3.6× speedup',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '32K Context (Medium)', lines: [
      { t: 'KV Cache = 4.0 GB!', mono: true, fill: D.orange.dark, fw: 700 },
      'KV cache equals 4-bit',
      'weights (50% of traffic)!',
      'Total: 8.0 GB / token',
      'Decode rate drops ~44%',
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: '128K Context (Long)', lines: [
      { t: 'KV Cache = 16.0 GB!', mono: true, fill: D.red.dark, fw: 700 },
      '80% of memory traffic!',
      'Total: 20.0 GB (> 16 GB',
      'RTX 5070 Ti → OOM!)',
      '4-bit gives only 1.6×',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Per-Token BF16 KV Size:  2 · L · H_kv · d_h · 2 B = 2 · 32 · 8 · 128 · 2 = 128 KiB/token',
      line2: 'For long-context workloads (32K–128K), KV cache dominates both capacity and decode bandwidth!' }
  ),

  'ch39-tools-5070ti-vs-3090': () => twoCols(
    'Entry Discrete Training Cards (Nov 2026): New 5070 Ti vs. Used 3090',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'RTX 5070 Ti (New, ~$1,100–1,300)', lines: [
      { t: '16 GB GDDR7  ·  896 GB/s  ·  300 W', mono: true, fill: D.blue.dark, fw: 700 },
      'Blackwell Tensor Cores: BF16, FP8, FP4',
      'Runs all D2L notebooks (peak < 13 GiB)',
      'Runs 7B–8B QLoRA fine-tunes easily',
      'Full retail warranty & modern efficiency',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'RTX 3090 (Used, ~$1,250–1,450)', lines: [
      { t: '24 GB GDDR6X  ·  936 GB/s  ·  350 W', mono: true, fill: D.orange.dark, fw: 700 },
      'Ampere Tensor Cores: BF16/FP16 (no FP8/4)',
      '+8 GB extra VRAM for larger batches/models',
      'No manufacturer warranty; higher heat/power',
      'Last consumer card with 2-way NVLink bridge',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: '16 GB is the practical floor for local deep learning; 12 GB excludes many 7B QLoRA runs.',
      line2: 'Budget for system RAM too: 64 GB of DDR5 (~$1,000–1,200 in late 2026) costs as much as the card.' }
  ),

  'ch39-tools-unified-memory-tiers': () => threeCols(
    'Unified-Memory Local Inference Systems (Nov 2026 Snapshot)',
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'AMD Strix Halo', lines: [
      { t: '128 GB  ·  256 GB/s', mono: true, fill: D.orange.dark, fw: 700 },
      '~$3,400–4,400 mini-PC',
      'Linux ROCm / Vulkan',
      '~50 tok/s on 120B MoE',
      '~1,000 tok/s prefill',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'NVIDIA DGX Spark', lines: [
      { t: '64/128 GB · 273 GB/s', mono: true, fill: D.blue.dark, fw: 700 },
      '$5,000 / $6,950 (GB10)',
      'Native CUDA Blackwell',
      '~60 tok/s on 120B MoE',
      { t: '2,400 tok/s prefill!', fill: D.blue.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Mac Studio M5 Max/Ultra', lines: [
      { t: '128–512 GB · 614–1200 GB/s', mono: true, fs: 11.2, fill: D.green.dark, fw: 700 },
      'From $2,500 / $5,500',
      'Apple MLX & llama.cpp',
      { t: '115–120 tok/s (Ultra)', fill: D.green.dark, fw: 700 },
      'Highest local RAM & BW',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Architectural Trade: Wide LPDDR5X gives 4–16× more capacity than discrete GDDR7,',
      line2: 'letting 70B–120B+ models run at home at moderate bandwidth-bound token rates.' }
  ),

  'ch39-tools-dense-vs-moe-decode': () => twoCols(
    'Why Mixture-of-Experts (MoE) Shines on Unified-Memory Systems',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Dense 70B Model (4-Bit ≈ 40 GB)', lines: [
      'Every token activates 100% of parameters',
      { t: 'Reads all ~40 GB of weights per token!', mono: true, fill: D.red.dark, fw: 700 },
      'Strix Halo / DGX Spark (260 GB/s):',
      { t: 'Decode ≲ 260 / 40 ≈ 6.5 tok/s (Slow!)', mono: true, fill: D.red.dark, fw: 700 },
      'M5 Ultra (1200 GB/s): ≲ 30 tok/s',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'MoE gpt-oss-120b (117B Total, 5.1B Active)', lines: [
      'Needs 59 GiB RAM to hold all 128 experts',
      'Routes each token to only 4 of 128 experts!',
      { t: 'Reads only ~3.2 GB of weights per token!', mono: true, fill: D.green.dark, fw: 700 },
      'Strix Halo / Spark: 50–60 tok/s (9× faster!)',
      'M5 Ultra (1200 GB/s): 115–120 tok/s!',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'MoE Decouples Capacity from Decode Bandwidth: Total parameters set the RAM required to fit,',
      line2: 'while active parameters per token set the bytes streamed per decode step!' }
  ),

  'ch39-tools-spark-vs-strix-prefill': () => twoCols(
    'Equal Decode Bandwidth (~265 GB/s), 2.4× Prefill Gap: Spark vs. Strix',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Short Chat (500-tok Prompt + 500 Reply)', lines: [
      'DGX Spark: 0.21s prefill + 8.33s decode = 8.5s',
      'Strix Halo: 0.50s prefill + 10.0s decode = 10.5s',
      { t: 'Decode dominates → only 1.2× difference', fill: D.blue.dark, fw: 700 },
      'Strix Halo delivers similar chat feel',
      'at roughly half the purchase price',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Coding Agent (30,000-tok Repo + 500 Reply)', lines: [
      { t: 'DGX Spark TTFT: 30,000 / 2400 = 12.5 s', mono: true, fill: D.green.dark, fw: 700 },
      { t: 'Strix Halo TTFT: 30,000 / 1000 = 30.0 s', mono: true, fill: D.red.dark, fw: 700 },
      'Total time: 20.8 s (Spark) vs. 40.0 s (Strix)',
      { t: '2.4× faster TTFT & 1.9× faster total!', fill: D.green.dark, fw: 700 },
      'Blackwell tensor cores + CUDA justify cost',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Why MoE Prefill Needs Large Chunks (≥ 512–2,048 tokens):',
      line2: 'A 512-token chunk touches ~126 of 128 experts/layer, amortizing the full 59 GiB weight read across tokens.' }
  ),

  'ch39-tools-calibrated-decode-fit': () => twoCols(
    'Auditing Hardware Claims with the Calibrated Affine Decode Model',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Physics Sanity Check (Reject Fake Blogs)', lines: [
      'Claim: 900 GB/s card decodes dense 70B',
      '4-bit (35 GB weights) at 45 tok/s?',
      { t: 'Implied BW = 45 × 35 = 1,575 GB/s!', mono: true, fill: D.red.dark, fw: 700 },
      '1,575 GB/s > 900 GB/s physical bus!',
      'Impossible without speculative/sparse',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Calibrated Affine Roofline Fit', lines: [
      { t: '1 / rate = t₀ + bytes / (η · β)', mono: true, fill: D.green.dark, fw: 700 },
      'η ∈ [0.60, 0.85]: achieved BW efficiency',
      't₀: fixed per-token dispatch overhead',
      'Fit on 6 llama-bench runs predicts new',
      'models & 16K contexts within ~10–15%!',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Trusted Primary Sources:  llama.cpp GitHub Discussions (llama-bench)  ·  ServeTheHome  ·  Level1Techs',
      line2: 'Always date street prices and verify reported token rates against β / B_active.' }
  ),

  // ── §39.5 software-ecosystem.md (10 diagrams) ──────────────────────────
  'ch39-tools-artifact-supply-chain': () => threeCols(
    'Treating External Models & Datasets as Supply-Chain Dependencies',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Evidence & Fit', lines: [
      'Filter 3M+ Hub models',
      'Triangulate leaderboards',
      'Verify on YOUR domain',
      'eval set with paired',
      'standard error bars!',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '2. Immutable Identity', lines: [
      'Pin 40-char Git SHA',
      '(never floating "main")',
      'Lock config, tokenizer,',
      'weights & parent SHA',
      'in a frozen manifest',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Safety & License', lines: [
      'Require safetensors',
      '(no pickle __reduce__)',
      'Audit trust_remote_code',
      'Record license at the',
      'exact pinned commit!',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Without an immutable revision, parent lineage, and domain eval, a benchmark cannot be reproduced.',
      line2: 'Without safetensors and remote-code review, loading a checkpoint can execute arbitrary code.' },
    true
  ),

  'ch39-tools-model-hubs-map': () => grid2x2(
    'Where the Open-Weight Ecosystem Lives (Nov 2026 Snapshot)',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Hugging Face Hub (3,000,000+ Models)', lines: [
      'Git-backed repos, model cards & safetensors',
      'Native transformers / diffusers / vLLM loading',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'ModelScope (280,000+ Models)', lines: [
      "Alibaba's hub for the Chinese open ecosystem",
      'First-party Qwen & multimodal releases',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Kaggle Models & Ollama Library', lines: [
      'Kaggle: zero-download /kaggle/input mounts',
      'Ollama: curated GGUF + chat-template bundles',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: 'Specialized: timm & Civitai', lines: [
      'timm: standardized PyTorch vision backbones',
      'Civitai: community diffusion weights & LoRAs',
    ]},
    { fill: C.lgray, st: C.gray, dk: C.ink, text: 'Download count ≠ task suitability: filter by architecture, license, recency, and model card.' }
  ),

  'ch39-tools-pretraining-lineage': () => threeCols(
    'Web-Scale Pretraining Corpus Lineage: Raw Crawl to FineWeb',
    { fill: C.lgray, st: C.gray, dk: C.ink, title: '1. Raw Common Crawl', lines: [
      { t: 'Petabytes of WARC HTML', mono: true, fill: C.ink, fw: 700 },
      'Unfiltered web dumps:',
      'boilerplate, spam,',
      'near-duplicates, PII,',
      'and benchmark leaks',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '2. C4 (2019, T5 Baseline)', lines: [
      { t: 'Colossal Clean Crawled', mono: true, fill: D.blue.dark, fw: 700 },
      'Heuristic deduplication,',
      'language ID & bad-word',
      'filtering; served as the',
      'first standard open corpus',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. FineWeb & FineWeb2', lines: [
      { t: '15T → 18.5T+ Tokens', mono: true, fill: D.green.dark, fw: 700 },
      'MinHash dedup + quality',
      'classifiers; FineWeb2 spans',
      '1,000+ languages (DataComp',
      'for image–text pairs)',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Datasets evolve across versions just like code (e.g. FineWeb v1.3.0 branch).',
      line2: 'Always pin the dataset commit hash and inspect its card for license and filtering biases.' },
    true
  ),

  'ch39-tools-papers-to-code': () => threeCols(
    'From Preprint to Verified Code: Avoiding Single-Index Lock-In',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Preprint & Curation', lines: [
      { t: 'arXiv (cs.LG / CL / CV)', mono: true, fill: D.blue.dark, fw: 700 },
      'Daily flood of papers →',
      'filtered via Hugging Face',
      'Papers, Semantic Scholar',
      '& alphaXiv discussions',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. Index Lesson (2025)', lines: [
      { t: 'Papers with Code Closed', mono: true, fill: D.orange.dark, fw: 700 },
      'Shut down July 2025;',
      'now redirects to HF',
      'Papers with a frozen',
      'GitHub data archive',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Primary Evidence', lines: [
      { t: 'GitHub + Hub Weights', mono: true, fill: D.green.dark, fw: 700 },
      'Check issue history,',
      'commit activity, and',
      'independent third-party',
      'reproduction logs',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Engineering Rule: Never make your build or benchmark pipeline depend on one external index.',
      line2: 'Archive the exact evaluation scripts, prompts, and baseline scores inside your own repository.' },
    true
  ),

  'ch39-tools-benchmark-decay': () => threeCols(
    "Why Static Leaderboards Decay & Retire (Goodhart's Law)",
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: '1. Contamination', lines: [
      'Static test questions or',
      'paraphrases leak into',
      'web pretraining crawls',
      'or synthetic post-train',
      'distillation datasets',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. Selection Overfit', lines: [
      'Labs evaluate 1,000s of',
      'checkpoint merges & RL',
      'hyperparameters against',
      'the public test split,',
      'overfitting its quirks',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '3. Ceiling & Test Flaws', lines: [
      'Open LLM Leaderboard',
      'retired in 2025;',
      'SWE-bench Verified',
      'replaced by SWE-bench',
      'Pro in Feb 2026',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Once a static benchmark becomes a marketing target, it ceases to measure generalization.',
      line2: 'Combine contamination-resistant rotating suites (LiveBench) with private domain evaluations.' },
    true
  ),

  'ch39-tools-five-evidence-types': () => grid2x2(
    'Triangulating Five Complementary Sources of Model Evidence',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Arena & 2. LiveBench', lines: [
      'Arena: blind human pairwise preference (Elo)',
      'LiveBench: rotated, withheld objective tasks',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Domain Capability Suites', lines: [
      'SWE-bench Pro (coding), math, long context',
      'Measures targeted structural/agentic skills',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '4. Artificial Analysis (Economics)', lines: [
      'Plots Quality vs. Latency (TTFT / tok/s)',
      'and Price ($/1M tokens) across providers',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '5. OpenRouter Token Volume', lines: [
      'Revealed preference of paying API customers',
      'Disagreements with Elo highlight cost/speed wins',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, text: 'Use public leaderboards to form a shortlist of 3–4 models, never as the final decision.' }
  ),

  'ch39-tools-arena-elo-resolution': () => twoCols(
    'Bradley-Terry Arena Elo: Vote Sample Size & Domain Reversals',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Statistical Resolution: SE ≈ 347 / √v', lines: [
      { t: 'P(A ≻ B) = 1 / [1 + 10^(−ΔR / 400)]', mono: true, fs: 11.8, fill: D.blue.dark, fw: 700 },
      'ΔR = +20 Elo → 52.9% win rate',
      'At v = 400 votes: SE ≈ 17.4 pts (Noise!)',
      'To resolve ΔR = 20 at 2·SE (SE ≤ 10):',
      { t: 'Requires v ≥ (347/10)² ≈ 1,204 votes!', mono: true, fill: D.blue.dark, fw: 700 },
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Simpson / Mixture Reversal on Your Task', lines: [
      'Global Arena: Model A leads B by +20 Elo',
      '(A wins 52.9% across all general prompts)',
      'On the 10% slice matching YOUR task:',
      { t: 'Model B wins 60% (A wins only 40%)!', mono: true, fill: D.red.dark, fw: 700 },
      '(A won 54.3% on the other 90% chat prompts)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Aggregate preference averages over prompt distributions that rarely match your workflow.',
      line2: 'A +20 Elo leaderboard lead can easily correspond to a 40%–60% loss on your actual task!' }
  ),

  'ch39-tools-domain-eval-pipeline': () => threeCols(
    'Three-Step Task-Specific Model Selection Pipeline',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Shortlist 3–4 Models', lines: [
      'Filter by license, size &',
      'VRAM/latency ceiling;',
      'pick top candidates from',
      'Arena, LiveBench &',
      'Artificial Analysis',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. Size Paired Eval Set', lines: [
      'Sample real domain data;',
      'size n via paired SE:',
      { t: 'SE = √[(p₁₀ + p₀₁) / n]', mono: true, fs: 11.5, fill: D.orange.dark, fw: 700 },
      'Include majority & simple',
      'heuristic baselines!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Paired Error Analysis', lines: [
      'Report scores ± SE and',
      'paired differences;',
      'inspect discordant examples',
      'where Model A ≠ Model B',
      'before shipping!',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Why Paired Comparisons Win: Evaluating all candidates on the exact same n test prompts',
      line2: 'cancels out prompt difficulty variance, needing far fewer examples than independent samples.' },
    true
  ),

  'ch39-tools-pickle-vs-safetensors': () => twoCols(
    'Checkpoint Deserialization Security: Legacy Pickle vs. Safetensors',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Legacy Pickle (.bin / .pt / .pkl)', lines: [
      'Stack-based VM executing pickle opcodes',
      { t: '__reduce__ invokes arbitrary callables!', mono: true, fill: D.red.dark, fw: 700 },
      'torch.load(..., weights_only=False) runs RCE',
      'Even weights_only=True opcode allowlist had',
      { t: 'bypass vulnerabilities (CVE-2025-32434)!', fill: D.red.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Safetensors (.safetensors Container)', lines: [
      { t: '[8-B len N][N-B UTF-8 JSON][Flat Bytes]', mono: true, fs: 11.5, fill: D.green.dark, fw: 700 },
      'JSON maps name → dtype, shape, [begin, end)',
      'Zero executable opcodes or callables!',
      'Validates non-overlapping bounds in O(tensors)',
      { t: 'Supports instant zero-copy OS mmap!', fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Supply-Chain Rule: Always pass allow_patterns=["*.json", "*.safetensors"] on download.',
      line2: 'Note: safetensors secures weight loading; running trust_remote_code=True still executes repo Python!' }
  ),

  'ch39-tools-remote-code-audit': () => threeCols(
    'Three Operational Controls for External Hub Dependencies',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: '1. Remote Code & auto_map', lines: [
      { t: 'trust_remote_code=True', mono: true, fs: 11.2, fill: D.red.dark, fw: 700 },
      'Executes repo .py files!',
      'If config.json auto_map',
      'points to another repo,',
      { t: 'pin BOTH commit SHAs!', fill: D.red.dark, fw: 700 },
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. License at Commit SHA', lines: [
      '"Open weights" spans:',
      '• Apache-2.0 / MIT',
      '• Custom MAU / use caps',
      '• Non-commercial CC-BY-NC',
      'Record license at SHA!',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '3. Cache Disk Hygiene', lines: [
      '~/.cache/huggingface',
      'accumulates 100s of GB:',
      { t: 'hf cache ls', mono: true, fill: D.blue.dark, fw: 700 },
      { t: 'hf cache rm <rev>', mono: true, fill: D.blue.dark, fw: 700 },
      { t: 'hf cache prune', mono: true, fill: D.blue.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Even when all repos are pinned to 40-char SHAs, outputs can still shift across machines',
      line2: 'due to kernel/cuBLAS float non-associativity, chat-template wrappers, or unseeded sampling.' }
  ),

  // ── §39.6 training-systems.md (10 diagrams) ─────────────────────────────
  'ch39-tools-7b-wall': () => twoCols(
    'Why a 7B Model Outgrows One GPU & What Distributed Training Splits',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Single-GPU Wall (7B Mixed-Precision Adam)', lines: [
      { t: 'BF16 Weights (2N):        14 GB', mono: true, fill: C.ink },
      { t: 'BF16 Gradients (2N):      14 GB', mono: true, fill: C.ink },
      { t: 'FP32 Master Weights (4N): 28 GB', mono: true, fill: C.ink },
      { t: 'FP32 Adam m & v (8N):     56 GB', mono: true, fill: D.red.dark, fw: 700 },
      { t: 'Total State = 112 GB > 80 GB H100!', mono: true, fill: D.red.dark, fw: 700 },
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Four Axes of Distributed Partitioning', lines: [
      '1. Batch Axis (DDP): split examples',
      '2. State Axis (FSDP / ZeRO): split 16N state',
      '3. Layer Axis (TP / PP): split mats or stages',
      '4. Sequence / Expert Axis (CP / EP)',
      { t: 'Each axis trades memory for collectives!', fill: D.blue.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Engineering Method: Start from a verified 1-GPU baseline (small proxy model if needed),',
      line2: 'and climb the scaling ladder one rung at a time as memory or wall-clock limits require.' }
  ),

  'ch39-tools-ddp-vs-fsdp2': () => twoCols(
    'Rung 2 vs. Rung 3: Replicated DDP vs. Sharded FSDP2 / NamedSharding',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Rung 2: Data Parallelism (DDP)', lines: [
      { t: 'State per GPU: 16N  (Replicated)', mono: true, fill: D.blue.dark, fw: 700 },
      'Every GPU holds full weights, grads, Adam',
      'Buckets & all_reduces gradients in backward',
      { t: 'Comm volume: 2(W−1)/W · S ≈ 2S', mono: true, fill: C.ink },
      'Best choice whenever 16N + acts fit!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Rung 3: FSDP2 (fully_shard / DTensor)', lines: [
      { t: 'State per GPU: 16N / W  (Sharded!)', mono: true, fill: D.green.dark, fw: 700 },
      'All-gathers layer weights just-in-time (fwd+bwd)',
      'Reduce-scatters gradients & frees non-owned',
      { t: 'Comm volume: 3(W−1)/W · S ≈ 3S (1.5×)', mono: true, fill: C.ink },
      '8× H100 shrinks 7B state: 84 → 10.5 GiB!',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'PyTorch FSDP2 represents sharded parameters as DTensor via fully_shard(module);',
      line2: 'JAX expresses the exact same sharding declaratively via NamedSharding(mesh, PartitionSpec).' }
  ),

  'ch39-tools-node-topology-mapping': () => twoCols(
    'Mapping 3D/4D Parallelism to Cluster Hardware (Ultra-Scale Playbook)',
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Intra-Node: Fast NVLink Domain (8 GPUs)', lines: [
      { t: 'Bandwidth: 450–900 GB/s per GPU', mono: true, fill: D.green.dark, fw: 700 },
      'Place Tensor Parallelism (TP ≤ 8) HERE!',
      'TP communicates 2× per transformer layer',
      'on the critical path (cannot be hidden)',
      'Also place Expert Parallel (EP) all-to-all here',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Inter-Node: InfiniBand / RoCE Network', lines: [
      { t: 'Bandwidth: 50–100 GB/s per NIC', mono: true, fill: D.orange.dark, fw: 700 },
      'Place Pipeline (PP) & Data/FSDP HERE!',
      'Stretching TP across 2 nodes costs −43%!',
      'Stretching PP across 2 nodes costs −14%',
      '(point-to-point only at stage boundaries)',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Topology Rule (4,000+ Measured Runs up to 512 GPUs):',
      line2: 'Keep highest-frequency collectives (TP) inside an NVLink node; scale PP and FSDP/DDP across nodes.' }
  ),

  'ch39-tools-gpipe-vs-1f1b': () => twoCols(
    'Pipeline Parallel Schedules across p Stages & m Microbatches',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'GPipe Schedule (All Fwd, Then All Bwd)', lines: [
      { t: '[F₁][F₂]…[F_m]  then  [B_m]…[B₂][B₁]', mono: true, fs: 11.8, fill: D.red.dark, fw: 700 },
      'Bubble fraction = (p − 1) / m',
      'Shrinking bubble needs m ≫ p (e.g. m = 32)',
      { t: 'Stage 1 holds ALL m activations at once!', fill: D.red.dark, fw: 700 },
      'O(m) activation memory causes OOM!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '1F1B Schedule (Warmup, Then Alternate)', lines: [
      { t: '[F₁]..[F_p] then [B₁][F_{p+1}][B₂][F_{p+2}]…', mono: true, fs: 11.5, fill: D.green.dark, fw: 700 },
      'Same bubble fraction = (p − 1) / m',
      'Frees microbatch j activation right after B_j!',
      { t: 'Caps live activations at ≤ p microbatches!', fill: D.green.dark, fw: 700 },
      'Interleaved v chunks cuts bubble to (p−1)/(mv)',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'At Fixed Global Batch B_global = 1,024:  m = 1,024 / (d · b) shrinks as DP degree d grows!',
      line2: 'At p = 8, b = 1: d = 8 gives m = 128 (5.5% bubble), whereas d = 64 gives m = 16 (43.8% bubble!).' }
  ),

  'ch39-tools-global-batch-invariant': () => twoCols(
    'Preserving the Global Batch Invariant When Scaling GPU Count',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Fixed Global Batch (Exact Equivalence)', lines: [
      { t: 'B_global = B_device · N_ranks · N_acc', mono: true, fill: D.blue.dark, fw: 700 },
      'Target: 1,024 seqs (4.2M tok), B_device = 4:',
      '• 32 GPUs → N_acc = 8   (2 step time)',
      '• 64 GPUs → N_acc = 4   (1 step time)',
      '• 256 GPUs → N_acc = 1  (Max DP world!)',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'What Breaks If You Forget to Adjust N_acc', lines: [
      'Doubling N_ranks from 64 → 128 at N_acc = 4',
      { t: 'DOUBLES B_global (1,024 → 2,048 seqs)!', fill: D.red.dark, fw: 700 },
      'Halves optimizer steps for fixed token budget',
      'If keeping 2× batch: scale Adam LR by √2',
      'and define warmup in TOKENS, not steps!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Always log B_global and tokens consumed alongside step count.',
      line2: 'Communication floor: as N_acc → 1, un-overlapped allreduce t_ar caps max multi-GPU speedup.' }
  ),

  'ch39-tools-eval-sampler-pitfalls': () => twoCols(
    'Why Multi-GPU Validation Accuracy Drifts & How to Fix It',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Default DistributedSampler Padding Bug', lines: [
      'Eval set = 10,001 examples across 8 ranks',
      'Sampler pads to 8 × 1,251 = 10,008 indices',
      { t: 'Examples 0..6 are evaluated TWICE!', fill: D.red.dark, fw: 700 },
      'Or drop_last=True drops 1..7 examples!',
      'Averaging rank ratios (1/W)Σ(c_r/n_r) is biased',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Exact Unpadded Distributed Evaluation', lines: [
      '1. Split 10,001 examples without padding',
      '   (Rank 0 gets 1,251; Ranks 1..7 get 1,250)',
      '2. Each rank tallies [correct_r, count_r]',
      { t: '3. all_reduce(SUM) both integers once!', mono: true, fill: D.green.dark, fw: 700 },
      { t: 'Acc = (Σ correct_r) / (Σ count_r)  (Exact!)', mono: true, fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Never average per-rank accuracy percentages across unequal shards.',
      line2: 'Sum the numerator (correct predictions) and denominator (total examples) across ranks separately.' }
  ),

  'ch39-tools-five-memory-knobs': () => grid2x2(
    'Combining Orthogonal Memory-Reduction Techniques Deliberately',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. State Sharding (FSDP2 / ZeRO)', lines: [
      'Divides params, grads & Adam moments by W',
      'Cost: ~1.5× collective comm over NVLink/IB',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '2. Activation Checkpointing + Accumulation', lines: [
      'Checkpointing: acts O(L) → O(1) (+33% FLOPs)',
      'Accumulation: shrinks microbatch B_device',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '3. CPU / NVMe Offload (ZeRO-Offload)', lines: [
      'Moves FP32 master weights & Adam to host',
      'Cost: PCIe transfer + CPU DDR time/step',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '4. Adapter Tuning (LoRA / QLoRA)', lines: [
      'Freezes base weights (4-bit NF4 in QLoRA)',
      'Drops >98% of gradients & Adam moments!',
    ]},
    { fill: C.lgray, st: C.gray, dk: C.ink, text: 'Apply only the techniques needed to fit target VRAM—extra knobs add comm or compute overhead.' }
  ),

  'ch39-tools-zero-offload-pipeline': () => threeCols(
    'ZeRO-Offload Step Budget (7B Model, 50 GB/s PCIe, 200 GB/s CPU DDR)',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. PCIe Transfers', lines: [
      { t: '14 GB BF16 Grads → CPU', mono: true, fs: 11.5, fill: D.blue.dark, fw: 700 },
      '(14 / 50 = 0.28 s)',
      { t: '14 GB BF16 W ← CPU', mono: true, fs: 11.5, fill: D.blue.dark, fw: 700 },
      '(14 / 50 = 0.28 s)',
      { t: 'Total PCIe = 0.56 s', mono: true, fill: D.blue.dark, fw: 700 },
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. CPU Adam Update', lines: [
      'Reads/writes 28 B/param:',
      '2B g + 8B W₃₂ + 16B m,v',
      '+ 2B W₁₆ = 196 GB traffic',
      'At 200 GB/s CPU DDR:',
      { t: 'CPU Update = 0.98 s', mono: true, fill: D.orange.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Amortization', lines: [
      'Total offload = 1.54 s',
      'To keep < 20% of step',
      'at 400 TF/s (42 GF/tok):',
      { t: 'Need ≥ 58,700 tok/step', mono: true, fs: 11.5, fill: D.green.dark, fw: 700 },
      'or 1-step delayed update!',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'One-Step Delayed Update overlaps CPU Adam (step k) with GPU Forward/Backward (step k+1),',
      line2: 'hiding CPU compute completely whenever t_GPU ≥ 0.98 s at the cost of 1 step of weight staleness.' },
    true
  ),

  'ch39-tools-scale-tool-selector': () => threeRows(
    'Selecting the Simplest Distributed Training Stack by Scale (Nov 2026)',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1 GPU (≥ 8–16 GB)  →  1 Node (2–8 GPUs, ≤ 30B Full Fine-Tune)', lines: [
      '1 GPU LoRA/QLoRA: Unsloth or HF PEFT   ·   1 Node DDP/FSDP2: HF Accelerate, Axolotl, LLaMA-Factory',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Few Nodes (16–64 GPUs)  →  Many Nodes (64–512+ GPUs, 3D/4D Pretraining)', lines: [
      'Few Nodes: PyTorch FSDP2 + torchrun or DeepSpeed   ·   Many Nodes: TorchTitan, Megatron Core, MaxText',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'RL Post-Training (DPO / GRPO Policy + Rollout Generation Loops)', lines: [
      'Single Node: HF TRL   ·   Multi-Node Disaggregated Rollouts: veRL or OpenRLHF (FSDP2 + vLLM/SGLang)',
    ]}
  ),

  'ch39-tools-atomic-checkpoint': () => twoCols(
    'Surviving Multi-Week Cluster Runs: Complete & Atomic Checkpoints',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Bitwise-Resumable Checkpoint Bundle', lines: [
      '1. Sharded model weights & Adam moments',
      '2. LR scheduler state (step/warmup counter!)',
      '3. Global step & total tokens consumed',
      '4. Deterministic data-loader cursor (seed + idx)',
      '5. Per-rank PRNG states (Python/NumPy/CUDA)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Crash-Safe Atomic Rename Protocol', lines: [
      'In-place torch.save crash corrupts file!',
      { t: '1. Write to ckpt.tmp in same directory', mono: true, fs: 11.8, fill: C.ink },
      { t: '2. f.flush(); os.fsync(f.fileno())', mono: true, fs: 11.8, fill: C.ink },
      { t: '3. os.replace("ckpt.tmp", "ckpt.pt")', mono: true, fs: 11.8, fill: D.green.dark, fw: 700 },
      { t: '4. os.fsync(dir_fd)  (power-loss safe!)', mono: true, fs: 11.8, fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Drill Recovery Before Day 1: Kill at step 40, restore on fresh workers, and verify bitwise parity at step 64!',
      line2: 'Omitted scheduler or data cursor state never raises an error—it silently ruins the loss curve.' }
  ),

  // ── §39.7 model-serving.md (9 diagrams) ─────────────────────────────────
  'ch39-tools-serial-vs-batched-serving': () => twoCols(
    'Why Serving 100 Concurrent Users Requires Dynamic Batching',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Serial Queue (1 Request at a Time)', lines: [
      'Each request takes 2.0 s of batch-1 decode',
      '100th colleague waits > 200 seconds!',
      'Streams 16.4 GB weights 100 separate times',
      '(1,640 GB read from HBM; ALUs 98% idle)',
      { t: 'High latency AND low GPU utilization!', fill: D.red.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Batched Decode (b Requests Together)', lines: [
      'All b active requests share the weights B_w!',
      'Each decode step reads 16.4 GB weights ONCE',
      'plus each active request\'s KV cache (n·B_kv)',
      'Multiplies token throughput by ~30–50×',
      { t: 'Bounded only by VRAM & TPOT latency SLO', fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Step Time for b Sequences of Context n:   t(b) = (B_w + b · n · B_kv) / β',
      line2: 'Aslong as b·n·B_kv ≲ B_w, adding concurrent users is nearly free in step time!' }
  ),

  'ch39-tools-throughput-vs-goodput': () => twoCols(
    'Why Raw Throughput Lies Under Overload: Throughput vs. Goodput',
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Raw Throughput (Total Tokens / Sec)', lines: [
      'Counts every token emitted regardless of wait',
      'As arrival rate λ exceeds server saturation,',
      'queue depth explodes and batch b swells',
      { t: 'Raw tok/s stays pinned near 100% peak', fill: D.orange.dark, fw: 700 },
      'while users wait minutes for a reply!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Goodput (SLO-Compliant Requests / Sec)', lines: [
      { t: 'Requires TTFT ≤ 1.0s AND TPOT ≤ 50ms', mono: true, fill: D.green.dark, fw: 700 },
      'Rises linearly with λ up to saturation,',
      'then COLLAPSES toward 0 without admission',
      'control as queued requests miss TTFT!',
      { t: 'Goodput is the true production metric', fill: D.green.dark, fw: 700 },
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Protecting Goodput Beyond Saturation:',
      line2: 'Combine chunked prefill (protects TPOT), early TTFT admission rejection, and disconnect cancellation.' }
  ),

  'ch39-tools-serving-stack-selector': () => threeRows(
    'Matching the Inference Engine to the Serving Workload (Nov 2026)',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Local Single-User Assistant  ·  Offline Dataset Batch Scoring', lines: [
      'Local: Ollama or LM Studio (llama.cpp GGUF) / MLX   ·   Offline Batch: vLLM offline mode (max tok/$)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Single App API  ·  Multi-User Interactive Chat & Agent Service', lines: [
      'App API: vLLM behind auth/TLS proxy   ·   Multi-User / Agents: vLLM or SGLang (RadixAttention tree)',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: 'Datacenter-Scale Multi-Node NVIDIA Fleet', lines: [
      'TensorRT-LLM (max kernel tok/s) orchestrated by NVIDIA Dynamo (disaggregated prefill/decode pools)',
    ]}
  ),

  'ch39-tools-gguf-memory-budget': () => threeCols(
    'Local GGUF Quantization Budget on Qwen3-8B (8.2B Params + 32K KV)',
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Q4_K_M (~4.9 bits/w)', lines: [
      { t: 'Weights ≈ 5.0–5.2 GB', mono: true, fill: D.green.dark, fw: 700 },
      '+ 32K BF16 KV (4.7 GB)',
      '= 9.9 GB total footprint',
      'Fastest decode speed;',
      'fits 16 GB cards easily',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Q5_K_M (~5.7 bits/w)', lines: [
      { t: 'Weights ≈ 5.8 GB', mono: true, fill: D.blue.dark, fw: 700 },
      '+ 32K BF16 KV (4.7 GB)',
      '= 10.5 GB total footprint',
      'Near-lossless perplexity',
      'at ~15% lower tok/s',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Q8_0 (~8.5 bits/w)', lines: [
      { t: 'Weights ≈ 8.7 GB', mono: true, fill: D.orange.dark, fw: 700 },
      '+ 32K BF16 KV (4.7 GB)',
      '= 13.4 GB total footprint',
      '42% slower decode than',
      'Q4_K_M; tight on 16 GB',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Local Sizing Rule: Pick the largest model whose 4-bit weights + target KV cache fit in RAM,',
      line2: 'then step up bit width (Q4_K_M → Q5_K_M → Q8_0) only if memory and decode speed allow.' }
  ),

  'ch39-tools-batched-tpot-scaling': () => threeCols(
    'Batched Decode Roofline: Qwen3-8B (BF16) on H100 (β = 3.35 TB/s, n = 2048)',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Batch b = 1 (Idle)', lines: [
      { t: 'Weights: 16.4 GB', mono: true, fill: C.ink },
      { t: 'KV Cache: 0.30 GB', mono: true, fill: C.ink },
      { t: 'TPOT = 5.0 ms', mono: true, fill: D.blue.dark, fw: 700 },
      'Throughput: 201 tok/s',
      '(Weight-read dominated)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Batch b = 64 (Balanced)', lines: [
      { t: 'Weights: 16.4 GB', mono: true, fill: C.ink },
      { t: 'KV Cache: 19.3 GB', mono: true, fill: D.green.dark, fw: 700 },
      { t: 'TPOT = 10.7 ms', mono: true, fill: D.green.dark, fw: 700 },
      'Throughput: 6,000 tok/s!',
      '(30× higher for 2× TPOT)',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'SLO Limit & Ceiling', lines: [
      { t: '20 ms SLO → b_max = 167', mono: true, fs: 11.5, fill: D.orange.dark, fw: 700 },
      '(8,350 tok/s at 20 ms)',
      'As b → ∞, throughput →',
      { t: 'β / (n·B_kv) = 11,090/s', mono: true, fs: 11.5, fill: D.red.dark, fw: 700 },
      '(100% KV-cache bound!)',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Even at b = 256, arithmetic intensity is 256×16.4 GFLOPs / 93.7 GB ≈ 45 FLOP/B < 295 ridge!',
      line2: 'Batched decoding at n = 2,048 remains memory-bandwidth bound all the way to b = 256.' }
  ),

  'ch39-tools-server-engine-arch': () => grid2x2(
    'Server Inference Engines: From Single Node to Disaggregated Fleet',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'vLLM (PagedAttention Standard)', lines: [
      'Continuous batching + 16-tok paged KV blocks',
      'Chained block-hash prefix cache & OpenAI API',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'SGLang (RadixAttention Tree)', lines: [
      'Maintains KV cache in a dynamic radix tree',
      'Excels on multi-turn agents & branching prompts',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'TensorRT-LLM (NVIDIA Kernel Engine)', lines: [
      'Hardware-specific fused FP8/NVFP4 kernels',
      'Maximizes steady-state tok/s on fixed fleets',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: 'NVIDIA Dynamo (Fleet Orchestrator)', lines: [
      'Separates Prefill GPU pool from Decode GPU pool',
      'Routes requests by worker load & prefix-cache hit',
    ]},
    { fill: C.lgray, st: C.gray, dk: C.ink, text: 'Start with vLLM or SGLang on 1–8 GPUs; adopt Dynamo when scaling across multi-node pools.' }
  ),

  'ch39-tools-paged-attention-blocks': () => twoCols(
    'KV Cache Allocation on 72 GB H100 (55.6 GB Cache Budget, Avg n = 2048)',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Contiguous Reservation (max_len=32,768)', lines: [
      'Reserves 32,768 × 147.5 KB = 4.83 GB / seq',
      { t: 'Max Batch b = ⌊55.6 / 4.83⌋ = 11 seqs!', mono: true, fill: D.red.dark, fw: 700 },
      'Actual used cache at n=2048 is 0.302 GB/seq',
      { t: '93.7% of reserved KV VRAM sits EMPTY!', fill: D.red.dark, fw: 700 },
      'Memory binds at b = 11 (far below 167 SLO)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'PagedAttention (16-Token Virtual Blocks)', lines: [
      'Allocates 2.36 MB blocks on demand via table',
      'Internal waste ≤ 15 tokens/seq (< 0.7% waste!)',
      { t: 'Max Batch b = ⌊55.6 / 0.302⌋ = 184 seqs!', mono: true, fill: D.green.dark, fw: 700 },
      '16.7× higher concurrency on the same GPU!',
      'Now the 20 ms TPOT SLO (b = 167) binds first',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'PagedAttention applies OS virtual-memory paging to KV tensors (Kwon et al., 2023),',
      line2: 'eliminating external fragmentation and enabling copy-on-write prefix sharing across requests.' }
  ),

  'ch39-tools-speculative-decoding': () => threeCols(
    'Speculative Decoding: Draft γ Tokens, Verify in One Parallel Pass',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Draft Model q', lines: [
      { t: 'Cost fraction c = 0.05', mono: true, fill: D.blue.dark, fw: 700 },
      'Autoregressively proposes',
      'γ = 4 tokens:',
      'x₁, x₂, x₃, x₄',
      '(takes 4 × 0.05 = 0.20)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '2. Target Verify p', lines: [
      { t: '1 Parallel Target Step', mono: true, fill: D.green.dark, fw: 700 },
      'Scores all γ+1 positions;',
      'accepts with min(1, p/q),',
      'resamples 1st reject from',
      'residual max(0, p − q)',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '3. Exact Speedup', lines: [
      { t: 'E[tok] = (1−α^(γ+1))/(1−α)', mono: true, fs: 11, fill: D.orange.dark, fw: 700 },
      'At α = 0.8, γ = 4:',
      '3.36 tok / 1.20 steps',
      { t: '= 2.80× Speedup!', fill: D.orange.dark, fw: 700 },
      'Exact target dist p(x)!',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Caveat on Busy Servers: Scoring b(γ+1) tokens in parallel is free only while memory-bound!',
      line2: 'On H100 (ridge ≈ 295 tok), for γ = 4 any batch b > 295 / 5 = 59 becomes compute-bound.' },
    true
  ),

  'ch39-tools-chunked-prefill-ops': () => twoCols(
    'Production Serving Controls: Chunked Prefill & Admission Guardrails',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Chunked Prefill (Protects TPOT SLO)', lines: [
      'Unchunked 16K prompt prefill blocks active',
      'decode users for > 200 ms (TPOT spike!)',
      'Chunked prefill caps step budget (e.g. 2,048',
      'tokens), splitting long prompts across steps',
      { t: 'and piggybacking active decode tokens!', fill: D.blue.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Admission, Cancellation & Proxy Hygiene', lines: [
      '1. Reject arrivals early when queue > TTFT SLO',
      '2. Propagate HTTP client disconnects to abort',
      '   abandoned generations immediately',
      '3. Front engine with TLS + auth + rate proxy',
      '4. Co-pin (model SHA, tokenizer, quant, engine)',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Benchmark p50 and p95 TTFT/TPOT under realistic open-loop Poisson arrivals,',
      line2: 'and run a 4-check conformance test (determinism, max_tokens, stop, stream usage) on engine upgrades.' }
  ),

  // ── §39.8 developers-guide.md (10 diagrams) ─────────────────────────────
  'ch39-tools-ipynb-vs-markdown': () => twoCols(
    'Version Control at Scale: Raw .ipynb JSON vs. Single-Source Markdown',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Editing Raw .ipynb Files in Git', lines: [
      'Nested JSON mixes prose, code, base64 PNGs,',
      'execution counters, and editor metadata',
      'A 1-word prose fix + cell re-run modifies',
      '500+ lines of JSON across the file!',
      { t: 'Unreadable PR diffs & constant conflicts', fill: D.red.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'D2L Single-Source Markdown (chapter_*/*.md)', lines: [
      'Pure human-readable Markdown + code fences',
      '1-word edit produces a clean 1-line git diff',
      'Executed outputs live in outputs/<fw>/*.json',
      'and re-capture ONLY when code hashes change',
      { t: 'Scales cleanly across 600+ notebooks!', fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Golden Rule: Edit ONLY chapter_*/*.md (never _notebooks/, *.qmd, or d2l/*.py).',
      line2: 'Every other artifact in the repository is derived deterministically by the build.' }
  ),

  'ch39-tools-single-source-fanout': () => threeCols(
    'One Markdown Source File Generates Five Synchronized Artifacts',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Source & Output Store', lines: [
      { t: 'chapter_*/*.md', mono: true, fill: D.blue.dark, fw: 700 },
      '(Prose, %%tab code,',
      '#@save, <!-- slides -->)',
      '+ committed store:',
      { t: 'outputs/<fw>/*.json', mono: true, fill: D.purple.dark, fw: 700 },
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Build Preprocessing', lines: [
      { t: 'make html / lib / ...', mono: true, fill: D.orange.dark, fw: 700 },
      'Splits framework tabs,',
      'splices stored outputs,',
      'extracts #@save blocks',
      '& compiles slide decks',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '5 Derived Outputs', lines: [
      '1. HTML Website (.qmd)',
      '2. Framework PDFs',
      '3. _notebooks/<fw>/*.ipynb',
      '4. _slides/<fw>/*.html',
      '5. d2l/*.py library',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'A one-line fix in chapter_*/*.md automatically updates the book website, PDF,',
      line2: 'Colab/Kaggle notebooks, lecture slides, and the d2l Python package in one rebuild!' },
    true
  ),

  'ch39-tools-framework-tab-routing': () => threeCols(
    'How Tagged (%%tab) and Untagged Cells Route Into Framework Notebooks',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Source File (.md)', lines: [
      { t: 'Cell 1: %%tab pytorch', mono: true, fs: 11.5, fill: D.blue.dark },
      { t: 'Cell 2: %%tab jax', mono: true, fs: 11.5, fill: D.purple.dark },
      { t: 'Cell 3: (untagged!)', mono: true, fs: 11.5, fill: D.green.dark, fw: 700 },
      { t: 'Cell 4: %%tab pytorch', mono: true, fs: 11.5, fill: D.blue.dark },
      { t: 'Cell 5: (untagged!)', mono: true, fs: 11.5, fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Generated Notebooks', lines: [
      { t: 'PyTorch .ipynb:', mono: true, fill: D.blue.dark, fw: 700 },
      'Runs Cells [1, 3, 4, 5]',
      { t: 'JAX .ipynb:', mono: true, fill: D.purple.dark, fw: 700 },
      'Runs Cells [2, 3, 5]',
      '(0 TF/MXNet notebooks)',
    ]},
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Shared-Cell Hazard', lines: [
      'Editing untagged Cell 3',
      'modifies BOTH PyTorch',
      'and JAX notebooks!',
      'Removing %%tab pytorch',
      'from x[0]=10 crashes JAX',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Rule: Whenever you edit an untagged shared cell, you must re-execute and capture',
      line2: 'outputs for EVERY framework notebook generated by that source file!' },
    true
  ),

  'ch39-tools-cell-id-and-save': () => twoCols(
    'Permanent Cell IDs (#slug-k) and the #@save Library Extractor',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Permanent Cell IDs (add_cell_ids.py)', lines: [
      { t: '```{.python .input #sec-slug-1}', mono: true, fs: 11.5, fill: D.blue.dark, fw: 700 },
      '1. Keys committed outputs in outputs/<fw>/',
      '2. Anchors slide references (@sec-slug-1)',
      '3. Survives moving paragraphs or inserting',
      '   new cells earlier in the section!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '#@save Extractor (make lib)', lines: [
      'Scans all chapters in book order into d2l/*.py',
      { t: 'Last-Definition-Wins Rule:', fill: D.green.dark, fw: 700 },
      '• §2.4 defines CPU Trainer.prepare_batch',
      '• §5.7 redefines GPU Trainer.prepare_batch',
      'd2l/torch.py keeps §5.7; §2.4 runs its own!',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'If you edit any cell carrying #@save, always run "make lib" first,',
      line2: 'then run "make audit-outputs" to see which downstream chapters became stale.' }
  ),

  'ch39-tools-bootstrap-pipeline': () => threeCols(
    'From Fresh Git Clone to Rendered HTML Site Without a Local GPU',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. ./bootstrap.sh', lines: [
      'Installs & pulls Git LFS',
      '(committed plot SVGs/PNGs)',
      'Creates .venv-build with',
      'Quarto & Python build',
      'tools (one-time setup)',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. Splice & Audit', lines: [
      { t: 'gmake html (GNU ≥ 4.3)', mono: true, fs: 11.5, fill: D.orange.dark, fw: 700 },
      'Runs audit_outputs.py',
      'Splices outputs/<fw>/*.json',
      'into generated .qmd files',
      '(no GPU execution needed!)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Quarto Render', lines: [
      'Renders full interactive',
      'HTML site & Reveal.js',
      'slide decks locally in',
      'seconds for fast prose',
      'and layout review',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Decoupling Execution from Rendering: Committed outputs let any laptop build the whole book;',
      line2: 'only notebooks whose code cells you actually modified need re-execution!' },
    true
  ),

  'ch39-tools-output-fingerprint': () => threeCols(
    'Three Hashes Inside outputs/<fw>/<chap>/<stem>.json Detect Staleness',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Cell Prefix Hash', lines: [
      'SHA of Cell k code',
      '+ ALL earlier cells 1..k−1',
      '(trailing spaces stripped)',
      'Prose edits = 0 stale;',
      'Cell 3 edit → 3..n stale!',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. Framework Version', lines: [
      { t: 'torch==2.11.0', mono: true, fill: D.orange.dark, fw: 700 },
      'Strips local build tags',
      '(+cu128 vs. macOS CPU)',
      'so CPU laptops do not',
      'trigger false alarms',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '3. Transitive d2l.X Hash', lines: [
      'Hashes the #@save block',
      'of every d2l.X symbol',
      'used in the section +',
      'all d2l helpers it calls',
      'transitively!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark,
      line1: 'Audit Gate (tools/audit_outputs.py): Stale text output stops the build with an error;',
      line2: 'stale image or missing multi-GPU hardware prints a warning so CPU contributors can preview.' }
  ),

  'ch39-tools-prefix-vs-ast-staleness': () => twoCols(
    'Why Prefix Hashing Beats Static AST Variable-Name Dataflow',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Static AST Variable Dataflow Fails On…', lines: [
      '1. PRNG stream shift: adding 1 random draw',
      '   in Cell 3 changes Cell 8 dropout/init',
      '   without sharing any variable name!',
      '2. In-place weight / dict / attribute mutation',
      '3. Global state (torch.set_default_dtype, rcParams)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'D2L Prefix Rule (Cells k..n Marked Stale)', lines: [
      'Any edit to Cell k invalidates Cells k..n',
      'Recomputing Cell k requires running Cells 1..k−1',
      'anyway in a fresh kernel!',
      { t: '100% sound against hidden state channels', fill: D.green.dark, fw: 700 },
      'Zero runtime tracer overhead',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Because a fresh kernel must replay cells 1..k−1 to reach cell k, running through cell n',
      line2: 'costs almost no extra time while guaranteeing complete end-to-end consistency.' }
  ),

  'ch39-tools-git-pr-flow': () => threeCols(
    'The Contributor Git Workflow: Fork → Branch → Edit Source → PR',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Fork & Branch', lines: [
      'Fork smolix/d2l-neu',
      { t: 'git clone <your-fork>', mono: true, fs: 11.5 },
      { t: 'git switch -c fix-topic', mono: true, fs: 11.5, fill: D.blue.dark, fw: 700 },
      'Use 1 dedicated branch',
      'per logical change',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. Edit & Verify', lines: [
      'Edit chapter_*/*.md',
      '(or use VS Code sync ext)',
      'Run lint_source.py',
      'Execute & capture any',
      'touched code cells',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Commit & Open PR', lines: [
      { t: 'git diff  (verify source)', mono: true, fs: 11.5 },
      { t: 'git commit -m "..."', mono: true, fs: 11.5 },
      { t: 'git push -u origin ...', mono: true, fs: 11.5 },
      { t: 'gh pr create', mono: true, fs: 11.5, fill: D.green.dark, fw: 700 },
      '(or GitHub pencil icon)',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Keep pull requests small and focused on one section or concept.',
      line2: 'When guiding coding agents, review git diff to ensure they never weakened tests or edited _notebooks/.' },
    true
  ),

  'ch39-tools-four-step-verify': () => grid2x2(
    'Four-Step Verification Loop Before Submitting a Code Pull Request',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Source Linter (< 100 ms)', lines: [
      { t: 'python3 tools/lint_source.py --corpus', mono: true, fs: 11.2, fill: D.blue.dark, fw: 700 },
      'Checks tab/slide divs, cell IDs, labels & xrefs',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. Rebuild Library (If #@save Edited)', lines: [
      { t: 'gmake lib', mono: true, fill: D.orange.dark, fw: 700 },
      'Updates d2l/*.py before notebook execution',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Execute & Capture Each Framework', lines: [
      { t: 'gmake -B _notebooks/<fw>/<ch>/<f>.executed', mono: true, fs: 10.8, fill: D.green.dark, fw: 700 },
      { t: 'gmake capture-outputs FILES=<ch>/<f>.md', mono: true, fs: 11, fill: D.green.dark, fw: 700 },
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '4. Audit & Render HTML', lines: [
      { t: 'gmake audit-outputs && gmake html', mono: true, fs: 11.5, fill: D.purple.dark, fw: 700 },
      'Confirms 0 stale cells and clean Quarto build',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, text: 'The VS Code extension (.vscode-extension/) runs execution, capture, and source sync in one click.' }
  ),

  'ch39-tools-noisy-bisection': () => twoCols(
    'Finding a Fault Across 64 Commits When the Training Test Is Noisy',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Standard 1-Run git bisect Fails', lines: [
      'Good commit passes w/ p = 0.90; bad w/ q = 0.30',
      'Binary search over 64 = 2⁶ commits makes 6 steps',
      '1 wrong noisy run sends search to wrong half!',
      { t: 'Success prob ≈ 26% (blames wrong commit!)', fill: D.red.dark, fw: 700 },
      'Majority vote (m=7 runs/step) takes 42 runs',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Bayesian Bisection & Fano Lower Bound', lines: [
      'Maintains posterior P(K = k | runs) over k=1..64',
      'Updates via Bayes rule after every noisy test',
      'until max_k P(K = k) ≥ 0.95 (~22–25 runs!)',
      { t: 'Fano: N ≥ (6 − H_b(P_e) − P_e·log₂63) / C', mono: true, fs: 10.5, fill: D.green.dark, fw: 700 },
      'Approaches Shannon channel capacity limit C!',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Each stochastic training run is a noisy binary channel with capacity C ≈ 0.32 bits/run (at p=0.9, q=0.3).',
      line2: 'Extracting log₂(64) = 6 bits of fault location at 95% confidence requires at least ~17–22 runs by physics!' }
  ),

  // ── §39.9 utils.md (10 diagrams) ────────────────────────────────────────
  'ch39-tools-d2l-lib-extraction': () => threeCols(
    'How the d2l Package Is Extracted From Chapter & Utility Cells',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Chapter #@save Cells', lines: [
      'Core classes taught in',
      'chapters: Module, Trainer,',
      'DataModule, Classifier,',
      'GPT, ResNet, Benchmark,',
      'kernel_times, ...',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '2. §39.9 Utility Cells', lines: [
      'Plumbing in utils.md:',
      'save_hyperparameters,',
      'plot, ProgressBoard,',
      'plot_intervals, RL grids,',
      'FeatureCNN, custom Embed',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. Generated d2l/*.py', lines: [
      { t: 'd2l/torch.py', mono: true, fill: D.green.dark, fw: 700 },
      { t: 'd2l/jax.py', mono: true, fill: D.green.dark, fw: 700 },
      { t: 'd2l/tensorflow.py', mono: true, fill: C.muted },
      { t: 'd2l/mxnet.py', mono: true, fill: C.muted },
      'Zero hidden magic!',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Active Core vs. Legacy Helpers: PyTorch and JAX chapters use d2l.Trainer.fit(model, data);',
      line2: 'legacy functions (train_ch6, train_seq2seq, Animator) remain in §39.9 only for older TF/MXNet tabs.' },
    true
  ),

  'ch39-tools-plot-log-ticks': () => twoCols(
    'Publication Line Charts: d2l.plot & Narrow-Decade Log Tick Placement',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Flexible Curve Normalization (d2l.plot)', lines: [
      'Accepts 1D array or list of curves for X, Y',
      'If Y is None: plots X against index 0..n−1',
      'If len(X) == 1: broadcasts shared x-axis',
      'Cycles 4 base styles + C0..C9 distinct colors',
      'Renders crisp vector SVG figures by default',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Smart Log-Axis Ticks (set_log_ticks)', lines: [
      'Standard log axes show only 10^k powers,',
      'leaving <2-decade plots with just 1 tick!',
      { t: 'When hi < 100 · lo (< 2 decades):', fill: D.green.dark, fw: 700 },
      'Places major ticks at 1, 2, 5 × 10^k!',
      'Plain decimals in [10⁻⁴, 10⁵), sci outside',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Why 1–2–5 Log Ticks Matter in Chapter 13 & Scaling Plots:',
      line2: 'Roofline sweeps and batch-scaling plots spanning 2×–20× stay immediately readable without grid guessing.' }
  ),

  'ch39-tools-progressboard-axes': () => twoCols(
    'ProgressBoard: Dual-Axis Telemetry & Headless Build Capture',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Dual-Axis Layout (_plot)', lines: [
      'Left Y-Axis (ax): train_loss & val_loss',
      { t: 'Right Y-Axis (ax.twinx()): *_acc curves', mono: true, fill: D.blue.dark, fw: 700 },
      'Top horizontal legend (bbox_to_anchor)',
      'never occludes descending loss curves',
      '1-point series get marker="o" (clip_on=False)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Interactive vs. Build Capture (show)', lines: [
      'Interactive Jupyter: redraws display_id',
      'in-place after every k training batches',
      { t: 'Book Build (D2L_NB_CAPTURE="1"):', fill: D.green.dark, fw: 700 },
      'Skips all intermediate final=False calls!',
      'Renders 1 clean final SVG at end of fit()',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Gating on D2L_NB_CAPTURE=1 avoids hundreds of redundant Matplotlib draws per epoch',
      line2: 'during automated notebook execution and keeps committed outputs/<fw>/*.json compact.' }
  ),

  'ch39-tools-spatial-visualizers': () => threeCols(
    'Spatial Visualizers: 2D Optimization Contours, Image Galleries & Heatmaps',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'show_trace_2d(f, res)', lines: [
      'Auto-widens 100×100 grid',
      'around iterates (x₁, x₂)',
      { t: 'set_aspect("equal")', mono: true, fill: D.blue.dark, fw: 700 },
      'preserves true gradient',
      'orthogonality to contours!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'show_images(imgs, r, c)', lines: [
      'Converts PyTorch/JAX/TF',
      'tensors via d2l.numpy()',
      'Arranges r × c gallery',
      'with hidden axis spines',
      'and optional class titles',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'show_heatmaps(M, ...)', lines: [
      'Takes 4D (r, c, H, W)',
      'attention/kernel matrices',
      { t: 'Shared Normalize(min,max)', mono: true, fs: 11, fill: D.orange.dark, fw: 700 },
      'across all panels (or',
      'per-panel cbar="each")',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Shared Color Normalization in show_heatmaps prevents low-entropy attention heads',
      line2: 'from looking artificially sharp when compared side-by-side against peaked heads!' }
  ),

  'ch39-tools-plot-intervals-modes': () => twoCols(
    'Statistical Uncertainty Visualization: d2l.plot_intervals',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Mode 1: kind="band" (Continuous)', lines: [
      'Plots central curve mean(x) with translucent',
      'fill_between(x, lo, hi, alpha=0.2) envelope',
      'Validates invariant: lo ≤ mean ≤ hi everywhere',
      '(raises ValueError on inverted bounds; NaNs ok)',
      'Ideal for sample-size & time-series curves',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Mode 2: kind="errorbar" (Side-by-Side)', lines: [
      'Computes symmetric horizontal dodge offsets',
      'in linear OR logarithmic axis coordinates!',
      'Multiple estimators sit side-by-side at each x',
      'without overlapping vertical error bars',
      'Supports reference lines & secondary marks',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Auto-rotates categorical x-tick labels by 30° whenever len(x) > 3 to prevent text collisions.',
      line2: 'Used across the Probability, Evaluation, and Bandit chapters to report estimates with error bars.' }
  ),

  'ch39-tools-rl-seed-pipeline': () => threeCols(
    'Multi-Seed Reinforcement Learning Aggregation & Gridworld Plots',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. run_seeds & on_grid', lines: [
      'Runs train(seed) across',
      'K independent seeds;',
      'on_grid applies causal',
      'moving-average smoothing',
      '& interpolates to step grid',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '2. plot_curves (Bands)', lines: [
      'Plots seed MEDIAN curve',
      'with shaded uncertainty:',
      { t: 'band="iqr" (25%–75%)', mono: true, fs: 11.5, fill: D.green.dark, fw: 700 },
      { t: 'or band="range" (min–max)', mono: true, fs: 11.5, fill: C.ink },
      'Resists outlier RL seeds!',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark, title: '3. show_grid(desc, V, π)', lines: [
      'Renders MDP gridworlds:',
      'Blues / diverging RdBu',
      'cell shading for V(s),',
      'state numbers, and',
      'greedy policy arrows π(s)',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      line1: 'Why Causal Window Smoothing + Seed Medians?',
      line2: 'Episode lengths vary with policy skill; interpolating onto a common env-step grid prevents survival bias.' },
    true
  ),

  'ch39-tools-jax-rl-bucketing': () => twoCols(
    'Eliminating XLA Recompilation Storms in JAX Reinforcement Learning',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Zero-Recompile Inference (forward_np)', lines: [
      'RL loops query single NumPy states millions',
      'of times while weights update every step',
      'forward_np caches 1 compiled jax.jit function',
      'keyed by id(graph) in _forwards dictionary',
      { t: 'Reads live v.get_raw_value() leaves in O(1)!', fill: D.blue.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '2. Power-of-Two Batch Bucketing (padded)', lines: [
      'Variable episode lengths n trigger XLA re-jits!',
      { t: '_bucket(n) = 2^max(6, ⌈log₂ n⌉)  (≥ 64)', mono: true, fs: 11.8, fill: D.green.dark, fw: 700 },
      'Pads arrays to {64, 128, 256, 512, 1024}',
      'and returns binary float32 mask (1=real, 0=pad)',
      { t: 'At most 5 compiled XLA kernels for any n!', fill: D.green.dark, fw: 700 },
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'All lengths n ∈ [1, 64] map to bucket size 64 → 1 single XLA compilation!',
      line2: 'Multiplying per-sample losses by mask and dividing by mask.sum() keeps gradients exact.' }
  ),

  'ch39-tools-feature-cnn-arch': () => threeCols(
    'FeatureCNN: Standardized 128-D Perceptual Scoring on CIFAR-10',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. Input & Antialias', lines: [
      'SHA-1 verified CIFAR-10',
      'rescaled to [−1, 1];',
      'Bilinearly resizes any',
      'H × W generated image',
      'to 32 × 32 (antialiased)',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '2. 3-Stage Conv Trunk', lines: [
      'Widths [32 → 64 → 128]:',
      'Each stage runs',
      'Conv(3×3) → BatchNorm',
      '→ ReLU → MaxPool(2×2)',
      '(spatial 32 → 16 → 8 → 4)',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '3. 128-D Feature & Head', lines: [
      'Global Average Pool →',
      { t: 'cnn.features(x) ∈ ℝ¹²⁸', mono: true, fill: D.green.dark, fw: 700 },
      'Used for FID / IS scores',
      'in GAN, VAE & Diffusion',
      '(15-ep One-Cycle Adam)',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Isolated Augmentation RNG: train_feature_cnn uses a private local Generator / PRNGKey',
      line2: 'for horizontal flips and reflect shifts so training never perturbs the caller\'s global seed!' },
    true
  ),

  'ch39-tools-custom-vjp-embed': () => twoCols(
    'Why JAX Small-Vocab Embeddings Need a @jax.custom_vjp Matrix Product',
    { fill: D.red.tint, st: D.red.base, dk: D.red.dark, title: 'Default Scatter-Add Backward (Slow!)', lines: [
      'Forward table[X] is a fast parallel gather',
      'Backward adds grad[b,t] into row table[X[b,t]]',
      'via GPU atomic memory operations!',
      'In byte/char models (V ≤ 1024), thousands of',
      { t: '" " and "e" tokens stall on the SAME row!', fill: D.red.dark, fw: 700 },
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Custom VJP One-Hot GEMM (V ≤ 1,024)', lines: [
      { t: 'O = jax.nn.one_hot(X, V)  ∈ ℝ^(B×T×V)', mono: true, fs: 11.5, fill: D.green.dark, fw: 700 },
      { t: '∇_table = einsum("...v,...d->vd", O, grad)', mono: true, fs: 11.2, fill: D.green.dark, fw: 700 },
      'Replaces serialized atomic scatter contention',
      'with one dense parallel Tensor Core matmul!',
      'Falls back to nnx.Embed when V > 1,024',
    ]},
    { fill: C.lblue, st: C.blue, dk: D.blue.dark,
      line1: 'Why the V = 1,024 Threshold? At V = 256 (byte GPT in §13.8), one-hot GEMM is 10–30× faster;',
      line2: 'at V = 50,257 (BPE GPT), materializing O(BTV) one-hot floats would cost too much memory and FLOPs.' }
  ),

  'ch39-tools-atomic-data-cache': () => threeCols(
    'Checksummed Atomic Dataset Caching & Sequence Masking Plumbing',
    { fill: C.lblue, st: C.blue, dk: D.blue.dark, title: '1. SHA-1 Cache Check', lines: [
      'Looks up file in ../data',
      'Streams 1 MB chunks into',
      'hashlib.sha1(); skips',
      'network download if hash',
      'matches DATA_HUB!',
    ]},
    { fill: C.lgreen, st: C.green, dk: D.green.dark, title: '2. Atomic Download', lines: [
      'Streams to tempfile in',
      'target dir, then swaps',
      { t: 'via os.replace(tmp, dst)', mono: true, fs: 11.2, fill: D.green.dark, fw: 700 },
      'Writes .extracted marker',
      'after zip/tar unpack',
    ]},
    { fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: '3. Sequence Masking', lines: [
      { t: 'arange(L) < valid_len', mono: true, fs: 11.5, fill: D.purple.dark, fw: 700 },
      'Broadcasts (1, L) vs (B, 1)',
      'Zeroes padded tokens in',
      'MaskedSoftmaxCELoss &',
      'attention score masks',
    ]},
    { fill: C.lamber, st: C.amber, dk: D.orange.dark,
      line1: 'Atomic os.replace + .extracted markers ensure interrupted downloads or parallel workers',
      line2: 'never leave half-written archives or re-extract 50,000 image files on every run.' },
    true
  ),
};
