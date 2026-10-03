"""Standard-library boundary to the isolated, official SadTalker CLI.

This module deliberately imports no ComfyUI or SadTalker/ML dependencies.
"""
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import wave

PREPROCESS = ("crop", "extcrop", "resize", "full", "extfull")


def _nonempty_file(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        if "checkpoint" in label.lower():
            raise FileNotFoundError(
                f"Missing or empty {label}: {path.name}. "
                f"SadTalker requires 8 models in the worker checkpoints directory. "
                f"Run 'python setup_worker.py' to download all required model weights automatically."
            )
        raise FileNotFoundError(f"Missing or empty {label}: {path}")
    return path


def load_config(path):
    """Load local config, interpreting relative paths beside config.json.

    Keep the interpreter's symlink intact: resolving a venv's python symlink
    would silently select the base interpreter instead of the isolated venv.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing SadTalker config: {path}. "
            f"To configure the isolated worker and automatically download all required models, run: "
            f"python setup_worker.py --comfy-python <path-to-comfy-python>. "
            f"See custom_nodes/comfyui-sadtalker/GUIDE.md."
        )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid SadTalker config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("SadTalker config must be a JSON object")
    config = dict(config)
    for key in ("python", "sadtalker_dir", "ffmpeg_dir"):
        value = config.get(key)
        if key == "ffmpeg_dir" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"SadTalker config requires a nonempty {key} path")
        target = Path(value).expanduser()
        if not target.is_absolute():
            target = path.parent / target
        config[key] = os.path.abspath(target)
    _nonempty_file(config["python"], "worker Python interpreter")
    if not os.access(config["python"], os.X_OK):
        raise ValueError(f"Worker Python is not executable: {config['python']}")
    _nonempty_file(Path(config["sadtalker_dir"]) / "inference.py", "SadTalker inference.py")
    if config.get("ffmpeg_dir") and not Path(config["ffmpeg_dir"]).is_dir():
        raise FileNotFoundError(f"Missing ffmpeg_dir: {config['ffmpeg_dir']}")
    timeout = config.get("timeout_seconds", 1800)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be a finite positive number")
    config["timeout_seconds"] = timeout
    return config


def validate_options(size, preprocess, still, enhancer, batch_size, expression_scale, pose_style):
    if type(size) is not int or size not in (256, 512):
        raise ValueError("size must be 256 or 512")
    if preprocess not in PREPROCESS:
        raise ValueError(f"preprocess must be one of {PREPROCESS}")
    if type(still) is not bool:
        raise ValueError("still must be boolean")
    if enhancer not in ("none", "gfpgan"):
        raise ValueError("enhancer must be none or gfpgan")
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be an integer in [1, 64]")
    if isinstance(expression_scale, bool) or not isinstance(expression_scale, (int, float)) or not math.isfinite(expression_scale) or not 0 <= expression_scale <= 3:
        raise ValueError("expression_scale must be finite and in [0, 3]")
    if type(pose_style) is not int or not 0 <= pose_style <= 45:
        raise ValueError("pose_style must be an integer in [0, 45]")


def build_command(config, image, audio, output, size, preprocess, still, enhancer, batch_size, expression_scale, pose_style):
    """Return a shell-free argv for the v0.0.2 safetensors worker."""
    validate_options(size, preprocess, still, enhancer, batch_size, expression_scale, pose_style)
    image = _nonempty_file(image, "source image").absolute()
    audio = _nonempty_file(audio, "driven audio").absolute()
    repo = Path(config["sadtalker_dir"])
    _nonempty_file(repo / "checkpoints" / f"SadTalker_V0.0.2_{size}.safetensors", "SadTalker checkpoint")
    mapping = "mapping_00109-model.pth.tar" if "full" in preprocess else "mapping_00229-model.pth.tar"
    _nonempty_file(repo / "checkpoints" / mapping, "mapping checkpoint")
    command = [str(config["python"]), "-u", str(repo / "inference.py"),
               "--source_image", str(image), "--driven_audio", str(audio),
               "--result_dir", str(Path(output).absolute()), "--size", str(size),
               "--preprocess", preprocess, "--batch_size", str(batch_size),
               "--expression_scale", str(expression_scale), "--pose_style", str(pose_style)]
    if still:
        command.append("--still")
    if enhancer == "gfpgan":
        command.extend(["--enhancer", "gfpgan"])
    return command


def discover_video(output):
    """Never mistake nested renderer intermediates for the final muxed MP4."""
    files = [path for path in Path(output).iterdir()
             if path.suffix.lower() == ".mp4" and not path.is_symlink()
             and path.is_file() and path.stat().st_size > 0]
    if not files:
        raise RuntimeError("SadTalker produced no nonempty final top-level MP4")
    if len(files) != 1:
        raise RuntimeError("SadTalker produced multiple final MP4 files (ambiguous output)")
    return files[0]


def _log_tail(path):
    with Path(path).open("rb") as log:
        log.seek(0, os.SEEK_END)
        log.seek(max(0, log.tell() - 16384))
        return log.read().decode("utf-8", errors="replace")


def _stop_process(process):
    """Kill the isolated process group on POSIX, direct worker on Windows."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
    finally:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def run_worker(config, command, output, check_interrupted=None):
    """Run with a log, time limit, and optional ComfyUI cancellation callback."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "worker.log"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    if config.get("ffmpeg_dir"):
        env["PATH"] = str(config["ffmpeg_dir"]) + os.pathsep + env.get("PATH", "")
    process = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Command: {json.dumps(command)}\nWorking directory: {config['sadtalker_dir']}\n")
        log.flush()
        try:
            if check_interrupted:
                check_interrupted()
            process = subprocess.Popen(command, cwd=str(Path(config["sadtalker_dir"]).resolve()), env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       shell=False, start_new_session=(os.name == "posix"))
            deadline = time.monotonic() + config.get("timeout_seconds", 1800)
            while process.poll() is None:
                if check_interrupted:
                    check_interrupted()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"SadTalker timed out after {config['timeout_seconds']} seconds")
                time.sleep(0.1)
            if check_interrupted:
                check_interrupted()
            if process.returncode:
                raise RuntimeError(f"SadTalker worker exited with code {process.returncode}")
            return discover_video(output)
        except BaseException as exc:
            if process is not None:
                _stop_process(process)
            log.flush()
            detail = f"{exc}\nWorker log: {log_path}\n{_log_tail(log_path)}"
            # Preserve Comfy's interrupt exception so queue cancellation stays cancellation.
            if isinstance(exc, (OSError, RuntimeError)):
                raise RuntimeError(detail) from exc
            if hasattr(exc, "add_note"):
                exc.add_note(f"Worker log: {log_path}")
            raise


def write_pcm16_wav(path, channels, sample_rate):
    """Downmix channel-major float samples to mono, clipping to PCM16."""
    if type(sample_rate) is not int or not 1 <= sample_rate <= 384000:
        raise ValueError("Audio sample_rate must be an integer in [1, 384000]")
    if not channels or not channels[0] or any(len(channel) != len(channels[0]) for channel in channels):
        raise ValueError("Audio must contain nonempty, equal-length channels")
    pcm = bytearray()
    for frame in zip(*channels):
        if any(not math.isfinite(value) for value in frame):
            raise ValueError("Audio samples must be finite")
        value = max(-1.0, min(1.0, sum(value / len(frame) for value in frame)))
        pcm.extend(struct.pack("<h", round(value * (32768 if value < 0 else 32767))))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
