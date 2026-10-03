# ComfyUI SadTalker — Isolated Custom Node

Generate talking-head videos inside ComfyUI from a portrait image and a speech audio clip.
SadTalker runs in a **completely separate Python 3.10 virtualenv**, so it never conflicts with ComfyUI's own environment.

> **Tested on:** Ubuntu 24.04 + RTX 4090 (CUDA 12.1), ComfyUI with Python 3.12.
> Windows setup notes are included but marked as untested until a separate verification run completes.

---

## How it works

```
LoadImage  ──┐
              ├──► SadTalkerIsolated ──► video_path (STRING)
LoadAudio  ──┘           │
                         └── inline video preview in the node
```

The `SadTalkerIsolated` node shells out to a pinned SadTalker `inference.py` via a
subprocess. No SadTalker or large ML weights are loaded into the ComfyUI process.
The isolated worker is checked out at a pinned git revision for reproducibility.

---

## Requirements

| Component | Version |
|-----------|---------|
| Python (ComfyUI host) | 3.10 – 3.12 |
| Python (worker venv) | **exactly 3.10** |
| CUDA | 12.1 (cu121 wheels) |
| PyTorch (worker) | 2.1.2+cu121 |
| FFmpeg | ≥ 4.4 (must be on PATH or passed via `--ffmpeg-dir`) |
| git | any recent version |
| Disk | ≈ 8 GB (models + venv) |

---

## Installation

### Step 1 — Install the custom node

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/sonukkc1312/comfyui-sadtalker.git
# or symlink from wherever you cloned the repo
ln -s /path/to/comfyui-sadtalker ./comfyui-sadtalker
```

### Step 2 — Create a separate Python 3.10 virtualenv

> Do **not** use the same Python environment as ComfyUI.

**Linux / macOS**

```bash
python3.10 -m venv /opt/sadtalker-venv
source /opt/sadtalker-venv/bin/activate
# Confirm the version
python --version   # must print Python 3.10.x
```

**Windows (PowerShell)**

```powershell
py -3.10 -m venv C:\sadtalker-venv
C:\sadtalker-venv\Scripts\Activate.ps1
python --version   # must print Python 3.10.x
```

### Step 3 — Run the installer

The installer clones SadTalker at a pinned revision, installs CUDA wheels,
downloads all 8 model files, and runs a CUDA smoke-check.
It writes `config.json` **only** if every step succeeds.

**Linux / macOS**

```bash
# Make sure the worker venv is active
source /opt/sadtalker-venv/bin/activate

python /path/to/comfyui-sadtalker/setup_worker.py \
  --comfy-python /path/to/ComfyUI/venv/bin/python \
  --sadtalker-dir /opt/SadTalker \
  --config /path/to/comfyui-sadtalker/config.json
```

**Windows (PowerShell)**

```powershell
C:\sadtalker-venv\Scripts\Activate.ps1

python C:\comfyui-sadtalker\setup_worker.py `
  --comfy-python C:\ComfyUI\python_embeded\python.exe `
  --sadtalker-dir C:\SadTalker `
  --config C:\comfyui-sadtalker\config.json `
  --ffmpeg-dir C:\ffmpeg\bin
```

> On Windows, ComfyUI typically ships with an **embedded Python** at
> `python_embeded\python.exe`. Pass that path to `--comfy-python`.
> The worker venv must still be a separate Python 3.10 installation.

#### Optional flags

| Flag | Description |
|------|-------------|
| `--ffmpeg-dir /path/to/bin` | Directory containing `ffmpeg` and `ffprobe` binaries. Only needed if they are not on `PATH`. |
| `--force-config` | Overwrite an existing `config.json` that points to a different worker. |

#### What the installer does

1. Validates that the worker Python ≠ the ComfyUI Python and is Python 3.10.
2. Clones `https://github.com/OpenTalker/SadTalker` and checks out revision `cd4c0465`.
3. Installs `torch==2.1.2+cu121`, `torchvision==0.16.2`, `torchaudio==2.1.2` and all SadTalker dependencies in the worker venv.
4. Downloads 8 model files (~3 GB total) from official GitHub releases; retries on failure.
5. SHA-256s each model on first download and re-checks on subsequent runs to detect corruption.
6. Runs a CUDA smoke-check (`torch.cuda.is_available()` + matmul).
7. Writes `config.json` only after all checks pass.

---

## Troubleshooting

### `CUDA unavailable` during smoke-check

- Confirm the GPU driver is installed: `nvidia-smi`
- Confirm the venv uses CUDA 12.1 wheels (not CPU-only): the installer uses `--index-url https://download.pytorch.org/whl/cu121`
- On rented cloud VMs, the driver may be present but `LD_LIBRARY_PATH` may be missing. Try:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```
  and re-run the installer.

### `Run this installer using the separate worker Python 3.10`

The installer detected a Python version other than 3.10 in the active venv.
Deactivate your current environment, activate the 3.10 venv, and retry.

### `Refusing to install into the ComfyUI Python environment`

You passed `--comfy-python` that resolves to the same prefix as the active
interpreter. Use a genuinely separate Python 3.10 venv.

### `basicsr` build fails with `no matching distribution found`

`basicsr==1.4.2` requires `--no-build-isolation` (the installer already passes this flag).
If it still fails, ensure `setuptools==69.5.1` and `wheel==0.43.0` are installed first —
the installer does this automatically in the correct order.

### `torchvision` or `torchaudio` version conflict

These are pinned to match `torch==2.1.2+cu121`. If any other package installs a
conflicting version, run `pip check` inside the worker venv; the installer calls this
automatically as the last step.

### FFmpeg not found

Install FFmpeg and add it to `PATH`, or use `--ffmpeg-dir /path/to/dir` containing
both `ffmpeg` and `ffprobe`.

**Linux:**
```bash
apt-get install ffmpeg
```

**Windows:**
Download a release build from <https://www.gyan.dev/ffmpeg/builds/> and pass
`--ffmpeg-dir C:\ffmpeg\bin`.

### `No such file or directory: config.json`

Run `setup_worker.py` first. The node requires `config.json` next to `__init__.py`.

### `SadTalker produced no nonempty final top-level MP4`

The worker ran but produced no output. Check `worker.log` in the run directory:
```
ComfyUI/output/sadtalker/<run-id>/worker.log
```
Common causes:
- No face detected in the portrait image — use a clear frontal face photo
- Audio file is too short (< 1 second) or has no voice
- Missing or corrupted model checkpoints — rerun `setup_worker.py`

### `SadTalker worker exited with code N`

Read the log tail printed with the error message. Common exit codes:
- **1** — Python traceback in the worker; see the log for the full trace
- **134** — SIGABRT — usually a CUDA out-of-memory error; try `size=256` or reduce `batch_size`

### `SadTalker timed out after 1800 seconds`

The worker exceeded the 30-minute timeout. Either:
- Reduce `size` from 512 to 256
- Reduce `batch_size`
- Increase `timeout_seconds` in `config.json`

---

## Usage

1. Start ComfyUI normally (`python main.py` or via the launcher).
2. Load the workflow from `workflows/sadtalker.json` (drag-and-drop into the canvas).
3. Upload your portrait image as `sadtalker-portrait.png` via the **LoadImage** node.
4. Upload your speech audio as `sadtalker-speech.wav` via the **LoadAudio** node.
5. Adjust the **SadTalkerIsolated** node parameters (see below).
6. Click **Queue Prompt**.

The generated MP4 will appear inline in the node as soon as inference finishes.
It is also saved under `ComfyUI/output/sadtalker/<run-id>/`.

### Node parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | — | Portrait photo (one face, frontal preferred) |
| `audio` | AUDIO | — | Speech audio (WAV or any format ComfyUI's LoadAudio supports) |
| `size` | 256 \| 512 | 256 | Resolution of the output face crop. 512 is slower and requires more VRAM |
| `preprocess` | crop \| extcrop \| resize \| full \| extfull | crop | How to align the face. `crop` is safest for most portraits |
| `still` | bool | false | Minimise head motion — best for podcast-style videos |
| `enhancer` | none \| gfpgan | none | Apply GFPGAN face enhancement. Adds latency; requires GFPGAN weights |
| `batch_size` | 1 – 64 | 2 | Internal frame batch size. Reduce to 1 if you hit OOM on 256 |
| `expression_scale` | 0.0 – 3.0 | 1.0 | Amplify or dampen facial expression |
| `pose_style` | 0 – 45 | 0 | Head pose style index from SadTalker's latent space |

### Recommended starting settings

| Use case | size | preprocess | still | enhancer | batch_size |
|----------|------|-----------|-------|----------|-----------|
| Quick test | 256 | crop | false | none | 2 |
| High quality | 512 | crop | false | gfpgan | 2 |
| Podcast / talking head | 256 | crop | true | none | 4 |
| Full body / wide shot | 256 | full | false | none | 2 |
| Low VRAM (< 8 GB) | 256 | crop | false | none | 1 |

---

## API workflow (headless)

Use `workflows/sadtalker-api.json` for the ComfyUI `/prompt` API or for batch automation.

Run the bundled acceptance tester to verify end-to-end:

```bash
# Place inputs in ComfyUI/input first
cp portrait.png /path/to/ComfyUI/input/sadtalker-portrait.png
cp speech.wav   /path/to/ComfyUI/input/sadtalker-speech.wav

python comfyui-sadtalker/test_workflow.py \
  --comfy-dir /path/to/ComfyUI \
  --evidence  /tmp/sadtalker-evidence \
  --base-url  http://127.0.0.1:8188
```

The tester queues the workflow, polls until complete, downloads the MP4, runs
`ffprobe` to confirm video+audio streams and duration, and writes a JSON evidence
file with all metadata.

---

## Re-running the installer

The installer is idempotent. It skips model downloads when the file already exists
and the SHA-256 matches the initially observed hash.  Use `--force-config` only if
you want to point the node at a different worker venv.

```bash
# Repair a broken venv (e.g. after a partial CUDA install)
source /opt/sadtalker-venv/bin/activate
python /path/to/comfyui-sadtalker/setup_worker.py \
  --comfy-python /path/to/ComfyUI/venv/bin/python \
  --sadtalker-dir /opt/SadTalker \
  --config /path/to/comfyui-sadtalker/config.json \
  --force-config
```

---

## Running the unit tests

```bash
# From the repository root — no GPU or ComfyUI installation required
python3 -m unittest discover -s comfyui-sadtalker/tests -v
```

All tests pass in any Python 3.10+ interpreter.  Tests that require `torch` or
`Pillow` are skipped automatically when those packages are absent.

---

## File layout

```
comfyui-sadtalker/
├── __init__.py            # ComfyUI entry point
├── nodes.py               # SadTalkerIsolated node class
├── runner.py              # Subprocess interface to SadTalker (no ML deps)
├── setup_worker.py        # One-time provisioning script
├── requirements-worker.txt# Pinned deps for the worker venv
├── config.example.json    # Template; setup_worker.py writes config.json
├── test_workflow.py        # End-to-end ComfyUI API acceptance tester
├── web/
│   └── preview.js         # ComfyUI JS extension — inline video preview
├── workflows/
│   ├── sadtalker.json     # GUI workflow (drag-and-drop into ComfyUI)
│   └── sadtalker-api.json # API-format workflow for /prompt endpoint
└── tests/
    ├── test_runner.py     # Unit tests for runner.py
    ├── test_setup.py      # Unit tests for setup_worker.py
    └── test_acceptance.py # Unit tests for test_workflow.py helpers
```

---

## Security notes

- Model hashes are **trust-on-first-observation**, not upstream-authenticated.
  The installer records the SHA-256 of each model on first download and re-checks on future runs.
- `config.json` is in `.gitignore` and should never be committed.
- The worker subprocess inherits a clean environment: `PYTHONPATH`, `PYTHONHOME` are
  removed, and only `ffmpeg_dir` is prepended to `PATH`.

---

## Known limitations

- Only the **first face** detected in the portrait is animated.
- Audio must contain voice; music-only or silent audio produces no output.
- `size=512` + `enhancer=gfpgan` requires ≈ 10 GB VRAM; reduce to `size=256` on 8 GB cards.
- Windows: FFmpeg must be provided via `--ffmpeg-dir`; it is not on the default PATH in most ComfyUI portable installs.
- `PYTHONPATH` and `PYTHONHOME` must be unset before running `setup_worker.py`.
