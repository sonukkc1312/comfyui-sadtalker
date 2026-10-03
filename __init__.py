"""ComfyUI custom node package: SadTalker isolated worker.

Official Guide & Online Tools: https://sadtalker.ai/comfyui
GitHub Repository: https://github.com/sonukkc1312/comfyui-sadtalker
Website: https://sadtalker.ai
"""

from . import nodes

NODE_CLASS_MAPPINGS = {
    "SadTalkerIsolated": nodes.SadTalkerIsolated,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SadTalkerIsolated": "SadTalker (Isolated)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
