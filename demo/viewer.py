"""ARDY's own 3D viewer (viser) with the SOMA skinned character, played from a background thread.

One viser server on a local port; the Gradio page embeds it in an iframe. ARDY's pinned viser fork installs
without its web client, so `ensure_client()` builds client/build/index.html with Node when it is missing."""
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

FPS = 30


def ensure_client():
    """Return None when viser's web client is built, else a message saying what is missing or wrong."""
    import viser
    root = Path(viser.__file__).resolve().parent / "client"
    if (root / "build" / "index.html").exists():
        return None
    node, npm, npx = shutil.which("node"), shutil.which("npm"), shutil.which("npx")
    if not (node and npm and npx):
        return ("The 3D viewer needs viser's web client, which is not built, and Node.js is not installed. "
                "Install Node.js 18 or newer from https://nodejs.org, then restart the app.")
    try:
        print("Building viser's web client (one time, a few minutes) ...", flush=True)
        subprocess.run("npm install --legacy-peer-deps", cwd=root, shell=True, check=True)
        subprocess.run("npx vite build --base ./ --outDir build", cwd=root, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return f"Building viser's web client failed ({e}). Run the two commands by hand in {root}."
    return None if (root / "build" / "index.html").exists() else f"The build finished but {root}/build/index.html is missing."


class Viewer:
    def __init__(self, port):
        import viser
        from ardy.skeleton import SOMASkeleton77
        from ardy.viz.viser_utils import Character

        self.port = port
        self.server = viser.ViserServer(host="127.0.0.1", port=port, label="EDDY")
        s = self.server
        s.scene.set_up_direction("+y")
        s.gui.configure_theme(show_logo=False, control_layout="floating", control_width="small", dark_mode=False)
        s.scene.add_grid("/grid", width=40.0, height=40.0,
                         wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
                         position=(0.0, 0.0001, 0.0), fade_distance=30.0, infinite_grid=True)
        self.char = Character("person", s, SOMASkeleton77(), create_skeleton_mesh=False, create_skinned_mesh=True,
                              visible_skeleton=False, visible_skinned_mesh=True, mesh_mode="soma_skin")
        self.motion = None
        self.frame = 0
        self.playing = False
        self.lock = threading.Lock()
        self._cam = None
        with s.gui.add_folder("Playback"):
            self.g_play = s.gui.add_button("Pause")
            self.g_frame = s.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0, disabled=True)
        self.g_play.on_click(lambda _: self.toggle())
        self.g_frame.on_update(self._on_slider)
        s.on_client_connect(self._on_connect)
        threading.Thread(target=self._loop, daemon=True).start()

    # --- control
    def load(self, joints, rots, contacts):
        """joints [T,77,3], rots [T,77,3,3], contacts [T,*] as numpy. Starts playing from frame 0."""
        j = torch.from_numpy(np.asarray(joints)).float()
        r = torch.from_numpy(np.asarray(rots)).float()
        c = torch.from_numpy(np.asarray(contacts)).float()
        xz = j[..., [0, 2]].reshape(-1, 2)
        lo, hi = xz.min(0).values, xz.max(0).values
        mid = (lo + hi) / 2
        j = j - torch.tensor([mid[0], 0.0, mid[1]])  # centre the path on the origin
        extent = float((hi - lo).max())
        self.char.precompute_skinning(j, r)
        with self.lock:
            self.motion = (j, r, c)
            self.frame = 0
            self.playing = True
        self.g_frame.max = max(1, len(j) - 1)
        self.g_frame.disabled = False
        self.g_play.label = "Pause"
        self._cam = (extent * 0.55 + 2.6)
        for cl in self.server.get_clients().values():
            self._frame_camera(cl)

    def toggle(self):
        with self.lock:
            self.playing = not self.playing
            if self.playing and self.motion is not None and self.frame >= len(self.motion[0]) - 1:
                self.frame = 0
        self.g_play.label = "Pause" if self.playing else "Play"
        return self.playing

    def _on_slider(self, ev):
        if self.motion is None or ev.client is None:  # ignore the playback loop's own updates
            return
        with self.lock:
            self.playing = False
            self.frame = int(self.g_frame.value)
        self.g_play.label = "Play"
        self._show(self.frame)

    def _frame_camera(self, client):
        d = self._cam or 5.0
        client.camera.position = (0.45 * d, 1.25 + 0.12 * d, 0.85 * d)
        client.camera.look_at = (0.0, 0.9, 0.0)

    def _on_connect(self, client):
        self._frame_camera(client)

    def _show(self, f):
        j, r, c = self.motion
        self.char.set_pose(j[f], r[f], c[f], frame_idx=f)

    def _loop(self):
        t_last = time.perf_counter()
        acc = 0.0
        while True:
            time.sleep(1 / 60)
            now = time.perf_counter()
            dt, t_last = now - t_last, now
            if not self.playing or self.motion is None:
                acc = 0.0
                continue
            acc += dt * FPS
            step = int(acc)
            if step < 1:
                continue
            acc -= step
            with self.lock:
                n = len(self.motion[0])
                self.frame += step
                if self.frame >= n:
                    self.frame = n - 1
                    self.playing = False
                    self.g_play.label = "Play"
                f = self.frame
            self._show(f)
            self.g_frame.value = f
