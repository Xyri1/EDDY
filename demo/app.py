"""EDDY demo: type a prompt, get an ARDY motion, with the text embedded by a remote Qwen3-Embedding-8B API.

    python demo/app.py [--adapter PATH] [--port 7860] [--viewer-port 8090]

No LLM2Vec encoder is loaded. Text -> your embedding endpoint -> EDDY (4096 -> 4096) -> ARDY -> motion.
The API key lives in this process only: it is never written to disk or logged."""
import argparse
import html
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root: eddy_encoder.py
sys.path.insert(0, str(HERE))

import gradio as gr
import numpy as np

from eddy_encoder import APIEmbedder, EddyTextEncoder

HF_REPO = "GhoXtie/Llama-3-EDDY-Qwen3-Embedding-8B"
HF_FILE = "eddy_qwen3-embedding-8b_4096.safetensors"
DEFAULT_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
KEY_ENV = "EDDY_DEMO_API_KEY"  # set from the page's password field, in memory only
ARDY_MODEL = "soma"

CLI = {"adapter": None, "viewer_port": 8090}
STATE = {"viewer": None, "viewer_error": None, "models": {}, "encoder": None, "encoder_key": None}
LOCK = threading.Lock()


# ---------------------------------------------------------------- helpers

def free_port(start):
    for p in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def pick_device(choice):
    import torch
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise gr.Error("CUDA is not available here. Choose cpu or auto.")
        return "cuda:0"
    return "cpu"


def esc(s):
    return html.escape(str(s))


def make_embedder(url, key, model):
    if not (key or "").strip():
        raise gr.Error("Enter your API key in Settings.")
    os.environ[KEY_ENV] = key.strip()
    # dims=4096 is checked on the response; the request does not ask for a size
    return APIEmbedder("api", model.strip(), 4096, url.strip(), key_env=KEY_ENV, send_dims=False)


def adapter_file(local):
    local = (local or "").strip() or CLI["adapter"]
    if local:
        p = Path(local).expanduser()
        if not p.is_file():
            raise gr.Error(f"Adapter file not found: {p}")
        return str(p)
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(HF_REPO, HF_FILE)
    except Exception as e:
        raise gr.Error(f"Could not download the adapter from {HF_REPO} ({type(e).__name__}). "
                       "Give a local adapter path in Settings, or start the app with --adapter.")


class TimedEmbedder:
    """Wraps the API embedder: records seconds spent in the endpoint. Same .meta."""

    def __init__(self, inner):
        self.inner, self.meta, self.seconds, self.calls = inner, inner.meta, 0.0, 0

    def __call__(self, texts):
        t = time.perf_counter()
        try:
            return self.inner(texts)
        finally:
            self.seconds += time.perf_counter() - t
            self.calls += 1


# ---------------------------------------------------------------- actions

def test_connection(url, key, model):
    emb = make_embedder(url, key, model)
    t = time.perf_counter()
    try:
        v = emb(["A person walks forward and waves with the right hand."])
    except AssertionError as e:
        return status("bad", "Wrong vector size", f"{esc(e)}. EDDY needs 4096. Use Qwen3-Embedding-8B at full size.")
    except Exception as e:
        return status("bad", "Connection failed", esc(str(e)[:300]))
    ms = (time.perf_counter() - t) * 1000
    return status("ok", f"Connected: 4096 dimensions in {ms:,.0f} ms",
                  "The size matches. That the endpoint really serves Qwen3-Embedding-8B is on you: "
                  "a different model with 4096 dimensions would pass this test and give bad motion.")


def status(kind, head, body=""):
    return f'<div class="note {kind}"><b>{head}</b><span>{body}</span></div>'


def get_viewer():
    return STATE["viewer"]


def get_ardy(device):
    if device not in STATE["models"]:
        from ardy.model import load_model
        STATE["models"][device] = load_model(ARDY_MODEL, device=device, text_encoder=False)
    return STATE["models"][device]


def get_encoder(url, key, model, adapter, device):
    k = (url, model, adapter, device, hash(key))
    if STATE["encoder_key"] != k:
        emb = TimedEmbedder(make_embedder(url, key, model))
        enc = EddyTextEncoder(emb, adapter_file(adapter), device=device)
        STATE["encoder"], STATE["encoder_key"] = enc, k
    return STATE["encoder"]


def generate(prompt, duration, seed, url, key, model, adapter, device_choice, progress=gr.Progress()):
    import torch
    from ardy.motion_rep.tools import length_to_mask
    from ardy.postprocess import post_process_motion
    from ardy.skeleton import SOMASkeleton30
    from ardy.tools import seed_everything, to_numpy

    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write a prompt first.")
    if STATE["viewer_error"]:
        gr.Warning(STATE["viewer_error"])
    duration = min(30.0, max(2.0, float(duration or 6)))
    device = pick_device(device_choice)
    with LOCK:
        progress(0.05, desc="Loading ARDY (first run downloads the weights)")
        t0 = time.perf_counter()
        model_ = get_ardy(device)
        load_s = time.perf_counter() - t0
        enc = get_encoder(url, key, model, adapter, device)
        model_.text_encoder = enc
        fps = model_.motion_rep.fps
        nf = int(duration * fps)

        progress(0.2, desc="Embedding the prompt")
        enc.embedder.seconds = 0.0
        enc.embedder.calls = 0
        t0 = time.perf_counter()
        enc([prompt])  # fills the encoder cache; generation reuses it
        embed_s = time.perf_counter() - t0
        api_s = enc.embedder.seconds

        progress(0.35, desc=f"Generating {duration:g} s of motion on {device}")
        seed_everything(int(seed))
        lengths = torch.tensor([nf], device=device)
        patch = model_.num_frames_per_token
        max_window = (int(10 * fps) // patch) * patch
        history = ((max_window - model_.gen_horizon_len) // patch) * patch
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            motion = model_([prompt], nf, num_denoising_steps=int(model_.diffusion.num_base_steps),
                            pad_mask=length_to_mask(lengths), first_heading_angle=torch.zeros(1, device=device),
                            motion_mask=None, observed_motion=None, cfg_weight=(2.0, 2.0),
                            crop_history_length=history)
            out = model_.motion_rep.inverse(motion, is_normalized=True)
        try:  # foot-skating cleanup needs ARDY's compiled motion_correction extension
            out.update(post_process_motion(out["local_rot_mats"], out["root_positions"], out["foot_contacts"],
                                           model_.skeleton, constraint_lst=None))
        except RuntimeError as e:
            if not STATE.get("pp_warned"):
                STATE["pp_warned"] = True
                print("Post-processing skipped:", str(e)[:120], flush=True)
                gr.Warning("Foot-skating cleanup is off: ARDY's motion_correction extension is not built. "
                           "The motion is the raw model output.")
        if isinstance(model_.skeleton, SOMASkeleton30):
            out = model_.skeleton.output_to_SOMASkeleton77(out)
        out = to_numpy(out)
        if device != "cpu":
            torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0

    arrays = {k: np.asarray(v[0]) if hasattr(v, "shape") and v.shape[:1] == (1,) else np.asarray(v)
              for k, v in out.items()}
    arrays["fps"] = np.asarray(fps)
    arrays["text"] = np.asarray(prompt)
    path = Path(tempfile.mkdtemp(prefix="eddy_")) / "motion.npz"
    np.savez(path, **arrays)

    viewer = get_viewer()
    if viewer is not None:
        viewer.load(arrays["posed_joints"], arrays["global_rot_mats"], arrays["foot_contacts"])
    name = torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace("GeForce ", "") if device != "cpu" else "CPU"
    stats = (f'<div class="stats">'
             f'<div><b>{api_s * 1000:,.0f} ms</b><span>embedding request</span></div>'
             f'<div><b>{embed_s * 1000 - api_s * 1000:,.0f} ms</b><span>EDDY adapter</span></div>'
             f'<div><b>{gen_s:,.1f} s</b><span>motion, {nf} frames</span></div>'
             f'<div><b>{esc(name)}</b><span>{"ARDY load " + format(load_s, ".1f") + " s" if load_s > 0.5 else "device"}</span></div>'
             f'</div>')
    return str(path), stats


def toggle_play():
    v = get_viewer()
    return v.toggle() if v else None


# ---------------------------------------------------------------- page

CSS = (HERE / "style.css").read_text(encoding="utf8")
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800'
        '&display=swap" rel="stylesheet">')

EXAMPLES = [
    "A person walks forward and waves with the right hand.",
    "A person crouches down and picks something up from the floor.",
    "A person jogs in a circle, then stops and stretches both arms overhead.",
    "A person throws a punch with the right hand and steps back.",
]


def build_ui():
    with gr.Blocks(title="EDDY demo") as ui:
        with gr.Row(elem_id="top"):
            gr.HTML('<h1>EDDY</h1><p class="lede">Text to motion with ARDY, without the 14 GB text encoder. '
                    'Your prompt goes to an embedding API, EDDY maps the vector, ARDY moves the body.</p>')
        with gr.Row(elem_id="main", equal_height=False):
            with gr.Column(scale=4, min_width=340, elem_id="side"):
                prompt = gr.Textbox(label="Prompt", lines=3, value=EXAMPLES[0], elem_id="prompt",
                                    placeholder="A person walks forward and waves with the right hand.")
                gr.HTML('<p class="hint">ARDY was trained on labels like the example: third person, one action '
                        'per event, joined with &ldquo;and&rdquo; or &ldquo;then&rdquo;.</p>')
                with gr.Row():
                    duration = gr.Number(value=6, minimum=2, maximum=30, step=0.5, label="Duration (s, 2 to 30)")
                    seed = gr.Number(value=0, precision=0, label="Seed", minimum=0)
                go = gr.Button("Generate motion", variant="primary", elem_id="go")
                with gr.Accordion("Settings", open=True, elem_id="settings"):
                    url = gr.Textbox(label="Embedding endpoint (OpenAI-compatible base URL)", value=DEFAULT_URL)
                    key = gr.Textbox(label="API key (kept in memory only)", type="password", value="")
                    model = gr.Textbox(label="Model name", value=DEFAULT_MODEL)
                    device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                    adapter = gr.Textbox(label="Adapter file (empty: download from Hugging Face)",
                                         value=CLI["adapter"] or "", placeholder=f"{HF_REPO}/{HF_FILE}")
                    test = gr.Button("Test connection", size="sm", elem_id="test")
                    test_out = gr.HTML()
            with gr.Column(scale=7, min_width=420, elem_id="stage"):
                v = STATE["viewer"]
                if v is not None:
                    gr.HTML(f'<iframe id="viewer" title="3D viewer" src="http://127.0.0.1:{v.port}/"></iframe>')
                else:
                    gr.HTML(f'<div class="novm">{esc(STATE["viewer_error"] or "The viewer did not start.")}</div>')
                with gr.Row(elem_id="bar"):
                    play = gr.Button("Play / pause", size="sm", scale=0, elem_id="play")
                    dl = gr.DownloadButton("Download .npz", size="sm", scale=0, elem_id="dl", value=None)
                stats = gr.HTML('<div class="stats idle"><div><b>-</b><span>embedding request</span></div>'
                                '<div><b>-</b><span>EDDY adapter</span></div>'
                                '<div><b>-</b><span>motion</span></div><div><b>-</b><span>device</span></div></div>')
        gr.HTML('<p class="built">Built with Meta Llama 3. Adapter: Llama-3-EDDY-Qwen3-Embedding-8B. '
                'Its training targets come from LLM2Vec, which runs Meta Llama 3 (Meta Llama 3 Community License).</p>')
        test.click(test_connection, [url, key, model], test_out)
        go.click(generate, [prompt, duration, seed, url, key, model, adapter, device],
                 [dl, stats])
        play.click(toggle_play, None, None)
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="local EDDY adapter (.safetensors); default: download from Hugging Face")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--viewer-port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    CLI["adapter"] = a.adapter
    from viewer import Viewer, ensure_client
    err = ensure_client()
    if err is None:
        STATE["viewer"] = Viewer(free_port(a.viewer_port))
    else:
        STATE["viewer_error"] = err
        print(err, flush=True)
    ui = build_ui()
    ui.queue().launch(show_error=True, server_name=a.host, server_port=a.port, css=CSS, head=HEAD,
                      theme=gr.themes.Base(), allowed_paths=[tempfile.gettempdir()])


if __name__ == "__main__":
    main()
