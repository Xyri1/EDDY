# EDDY demo

Built with Meta Llama 3. The model is Llama-3-EDDY-Qwen3-Embedding-8B (see [LICENSE-LLAMA3.txt](../LICENSE-LLAMA3.txt)); the page shows this note in its footer.

A local Gradio page: type a prompt, get an ARDY motion, see it in ARDY's own 3D viewer. The prompt is embedded by
a Qwen3-Embedding-8B API of your choice, EDDY maps the vector, ARDY generates. No LLM2Vec encoder (14 GB) is loaded.

## Install

Python 3.11.

```
pip install -r demo/requirements.txt
```

- Install a CUDA build of torch first for GPU use (see pytorch.org). The demo also runs on CPU.
- ARDY needs `numpy<2`. If another package pulled in numpy 2, run `pip install "numpy<2"`.
- ARDY's viser fork installs without its web client. At start the app checks `client/build/index.html` in the
  installed viser package. If it is missing and Node.js is installed, the app builds it (`npm install
  --legacy-peer-deps`, `npx vite build --base ./ --outDir build`; a few minutes, once). Without Node.js the
  page says to install it (https://nodejs.org, 18 or newer) and the viewer stays off.
- Weights: ARDY downloads `nvidia/ARDY-SOMA-RP-30FPS-Horizon60` from Hugging Face on the first generation (public).
  The EDDY adapter comes from `GhoXtie/Llama-3-EDDY-Qwen3-Embedding-8B`, file `eddy_qwen3-embedding-8b_4096.safetensors`.

## Run

Copy `.env.example` to `.env` in the repository root. Set `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_API_KEY` for your embedding provider. The demo loads this file at startup. Existing environment variables take precedence.

```
python demo/app.py                     # open http://127.0.0.1:7860
python demo/app.py --adapter path/to/eddy_qwen3-embedding-8b_4096.safetensors
```

Options: `--port` (7860), `--viewer-port` (8090, the viewer's local server; the next free port is used), `--host`.

## Settings

- **Embedding endpoint:** initialized from `EMBEDDING_BASE_URL`. Use an OpenAI-compatible base URL, without `/embeddings`. Defaults to `https://openrouter.ai/api/v1` if unset.
- **API key:** read from `EMBEDDING_API_KEY` in `.env` or the environment. Leave the Settings field empty to use it, or enter an override. The configured key stays on the server; the app never writes keys to disk or logs them.
- **Model name:** initialized from `EMBEDDING_MODEL`. Use your provider's model ID for Qwen3-Embedding-8B. Defaults to `qwen/qwen3-embedding-8b` if unset.
- **Device:** auto, cuda or cpu.
- **Adapter file:** a local path. Empty means download from Hugging Face.
- **Test connection:** embeds one prompt, reports the vector size (must be 4096) and the latency. It cannot tell
  whether the endpoint really serves Qwen3-Embedding-8B. That is your responsibility.

## Generate

Prompt, duration (2 to 30 s), seed. The page shows the motion, timings (embedding request, adapter, motion
generation, device) and offers the `.npz` (`posed_joints` [T,77,3], `global_rot_mats`, `foot_contacts`, `fps` 30, `text`).
The embedding time is 0 ms when the same prompt was embedded before (the encoder caches it).
Use SEED-style prompts: third person, one action per event ("A person walks forward and waves with the right hand.").

## Known limits

- Foot-skating cleanup needs ARDY's compiled `motion_correction` extension. Without it (as in the test setup) the
  app shows the raw model output and says so once.
- On a phone-width screen the viewer sits below the settings.

## Measured timings

Windows 11, RTX 4080 SUPER, Ryzen 9 7900X, torch 2.11, 6 s motion (180 frames), full ARDY schedule, warm model.
One run each after warm-up, seed 0, the prompt above:

| Device | Motion generation | First load of ARDY |
|---|---|---|
| GPU (cuda) | 0.4 to 0.7 s | about 10 to 14 s |
| CPU | 2.8 s | about 1.2 s (weights already on disk) |

Embedding request through OpenRouter: 1.4 to 2.2 s. EDDY adapter: under 0.1 s on GPU, a few ms on CPU.
These are single runs, not a benchmark.
