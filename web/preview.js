/**
 * SadTalker video preview extension for ComfyUI.
 *
 * Registers a nodeCreated hook that attaches a <video> DOM widget to any
 * SadTalkerIsolated node.  Each onExecuted call replaces the source in-place
 * so the widget survives workflow re-runs without accumulating DOM nodes.
 * onRemoved pauses and clears src to release the media handle immediately.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "SadTalker.preview",

  async nodeCreated(node) {
    if (node.comfyClass !== "SadTalkerIsolated") {
      return;
    }

    const video = document.createElement("video");
    video.controls = true;
    video.loop = false;
    video.style.width = "100%";
    video.style.maxHeight = "320px";
    video.style.objectFit = "contain";
    video.style.display = "block";
    video.style.background = "#000";

    const widget = node.addDOMWidget("preview", "video", video, {
      serialize: false,
      hideOnZoom: false,
    });

    node.onExecuted = function (message) {
      const videos = (message || {}).videos || [];
      if (!videos.length) return;

      const { filename, subfolder, type } = videos[0];
      const params = new URLSearchParams({ filename, subfolder, type });
      // Append a cache-buster so the browser always fetches the newest file.
      params.set("t", Date.now());
      video.src = api.apiURL("/view?" + params.toString());
      video.removeAttribute("hidden");
      video.load();

      // Resize the node to accommodate the video widget.
      const [w] = node.computeSize();
      node.setSize([Math.max(w, 320), 520]);
      node.setDirtyCanvas(true, true);
    };

    node.onRemoved = function () {
      video.pause();
      video.src = undefined;
      video.remove();
    };
  },
});
