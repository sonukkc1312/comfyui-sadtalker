"""ComfyUI custom node package: SadTalker isolated worker.

Install this directory as a ComfyUI custom node (symlink or copy under
ComfyUI/custom_nodes/). Before use, run setup_worker.py in a separate
Python 3.10 virtualenv to provision SadTalker and write config.json.
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
