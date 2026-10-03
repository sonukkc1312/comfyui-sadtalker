"""SadTalkerIsolated ComfyUI node.

Invokes SadTalker in a separate, pinned Python 3.10 environment through
runner.py.  Never imports SadTalker or ML packages directly — isolation is
the whole point.

Requires config.json in the same directory, written by setup_worker.py.
"""
import importlib.util
import os
from pathlib import Path
import uuid

# ---------------------------------------------------------------------------
# Locate runner.py without importing it at module scope as a sub-package.
# This allows tests to swap CONFIG_PATH via patch.object(nodes, "CONFIG_PATH").
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_RUNNER_SPEC = importlib.util.spec_from_file_location("sadtalker_runner", _HERE / "runner.py")
_runner_module = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_runner_module)

CONFIG_PATH = _HERE / "config.json"

# ---------------------------------------------------------------------------
# Input serialisation helpers
# ---------------------------------------------------------------------------

def _tensor_to_pil(image_tensor):
    """Convert a ComfyUI IMAGE tensor (B,H,W,C float32 0-1) to a PIL Image.

    Only the first frame of a batch is used.  Raises ValueError on bad shape
    or non-finite pixel values.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Pillow and numpy are required in the ComfyUI environment") from exc

    if (
        not hasattr(image_tensor, "shape")
        or image_tensor.ndim != 4
        or image_tensor.shape[0] != 1
        or image_tensor.shape[3] != 3
        or image_tensor.shape[1] < 1
        or image_tensor.shape[2] < 1
    ):
        raise ValueError(
            f"IMAGE tensor must be (1, H≥1, W≥1, 3), got {getattr(image_tensor, 'shape', None)}"
        )
    frame = image_tensor[0]  # (H, W, 3)
    # Check for non-finite values without importing torch explicitly.
    try:
        import math
        flat = frame.reshape(-1).tolist()
        if any(not math.isfinite(v) for v in flat):
            raise ValueError("IMAGE tensor contains non-finite values")
    except Exception as exc:
        if "non-finite" in str(exc):
            raise
    # Single-pass conversion: float32 [0,1] → uint8 [0,255].
    # Do NOT multiply by 255 twice — clamp, scale, round, convert.
    arr_np = (frame.clamp(0.0, 1.0).cpu().float().numpy() * 255).round().clip(0, 255).astype("uint8")
    return Image.fromarray(arr_np, mode="RGB")


def serialize_inputs(image_tensor, audio_dict, work_dir):
    """Write IMAGE + AUDIO tensors to PNG + WAV files inside work_dir.

    Returns (image_path, audio_path) as Path objects.

    Raises ValueError for invalid shapes/types, RuntimeError for missing
    ComfyUI-env dependencies.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Image ---
    pil = _tensor_to_pil(image_tensor)
    image_path = work_dir / "source.png"
    pil.save(str(image_path), format="PNG")

    # --- Audio ---
    if not isinstance(audio_dict, dict) or "waveform" not in audio_dict:
        raise ValueError("AUDIO input must be a dict with 'waveform' and 'sample_rate'")
    waveform = audio_dict["waveform"]  # (B, C, T) float32
    sample_rate = audio_dict.get("sample_rate")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError("Audio sample_rate must be an int")
    if (
        not hasattr(waveform, "shape")
        or waveform.ndim != 3
        or waveform.shape[0] != 1
        or waveform.shape[1] < 1
        or waveform.shape[2] < 1
    ):
        raise ValueError(
            f"AUDIO waveform must be (1, C≥1, T≥1), got {getattr(waveform, 'shape', None)}"
        )
    # Use only the first batch element; pass all channels to runner for mono downmix.
    channels = [waveform[0, c, :].tolist() for c in range(waveform.shape[1])]
    audio_path = work_dir / "driven.wav"
    _runner_module.write_pcm16_wav(audio_path, channels, sample_rate)
    return image_path, audio_path


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class SadTalkerIsolated:
    """Generate a talking-head video from a portrait image and a speech audio clip.

    SadTalker runs in a completely separate Python 3.10 virtualenv; this node
    never loads any SadTalker or large ML model into the ComfyUI process.

    Before using this node, run setup_worker.py to provision the worker
    environment and write config.json.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "video/sadtalker"
    DESCRIPTION = (
        "Generate talking-head videos from a portrait image and speech audio.\n"
        "Runs in an isolated Python 3.10 worker. Requires 8 model checkpoints.\n"
        "Run 'python setup_worker.py' to download all required weights automatically."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "size": ([256, 512], {"default": 256}),
                "preprocess": (["crop", "extcrop", "resize", "full", "extfull"], {"default": "crop"}),
                "still": ("BOOLEAN", {"default": False, "label_on": "still", "label_off": "animated"}),
                "enhancer": (["none", "gfpgan"], {"default": "none"}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "expression_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "pose_style": ("INT", {"default": 0, "min": 0, "max": 45, "step": 1}),
            }
        }

    def generate(self, image, audio, size, preprocess, still, enhancer,
                 batch_size, expression_scale, pose_style):
        import folder_paths
        import comfy.model_management as model_management

        output_root = Path(folder_paths.get_output_directory())
        # Each run gets a unique 32-char hex subdirectory so reruns never
        # overwrite each other and the UI can show both results.
        run_id = uuid.uuid4().hex
        run_dir = output_root / "sadtalker" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Serialize inputs directly into run_dir so they persist for the
        # full duration of the subprocess. Using a tempdir caused a race:
        # the tempdir was deleted as soon as the 'with' block exited, while
        # SadTalker was still reading the PNG/WAV files from it.
        image_path, audio_path = serialize_inputs(image, audio, run_dir)

        model_management.throw_exception_if_processing_interrupted()
        config = _runner_module.load_config(CONFIG_PATH)
        command = _runner_module.build_command(
            config, image_path, audio_path, run_dir,
            size, preprocess, still, enhancer,
            batch_size, expression_scale, pose_style,
        )
        video = _runner_module.run_worker(
            config, command, run_dir,
            check_interrupted=model_management.throw_exception_if_processing_interrupted,
        )

        # Build ComfyUI preview metadata.
        subfolder = str(video.parent.relative_to(output_root))
        metadata = {
            "filename": video.name,
            "subfolder": subfolder,
            "type": "output",
        }
        return {"ui": {"videos": [metadata]}, "result": (str(video),)}

