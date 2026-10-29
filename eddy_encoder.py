"""EDDY inference: adapter model, loading, and the drop-in text encoder for ARDY.

Needs only numpy, torch and safetensors. transformers is imported lazily by QwenEmbedder
(local Qwen embedding) only.

  EddyTextEncoder(embedder, adapter_path)  embedder: any callable texts -> [B, 4096]
  load_adapter(path)                       -> (model, meta)
  QwenEmbedder / APIEmbedder               ready-made embedders (local Qwen3-Embedding; OpenAI-compatible API)
"""

import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)  # a zero row stays zero


class QwenEmbedder:
    """Qwen3-Embedding as in its model card: left padding, last-token pooling,
    MRL truncation to `dims`, then L2 normalization. No instruction (documents are embedded raw).
    Used both to build training inputs and at serving time, so both go through the same code."""

    def __init__(self, path, dims=None, device="cuda", dtype="bfloat16", max_length=512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        name = Path(str(path)).name.lower()  # model ID name, for example qwen3-embedding-0.6b
        # path: a local folder or a Hugging Face ID, for example Qwen/Qwen3-Embedding-8B
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModel.from_pretrained(path, dtype=getattr(torch, dtype)).to(device).eval()
        self.dims = dims or self.model.config.hidden_size
        self.device, self.max_length = device, max_length
        self.meta = {"provider": "local", "model": name,
                     "dims": str(self.dims), "dtype": dtype, "instruction": "", "settings": "{}"}

    def __call__(self, texts):
        import torch
        import torch.nn.functional as F
        batch = self.tok(list(texts), padding=True, truncation=True,
                         max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            h = self.model(**batch).last_hidden_state[:, -1]  # left padding: last position is the last token
        return F.normalize(h[:, : self.dims].float(), p=2, dim=1).cpu().numpy()


class APIEmbedder:
    """Any OpenAI-compatible /embeddings endpoint (OpenAI, and the compatible modes of DashScope and
    Gemini; check each provider's docs for its base URL and limits). Stdlib only. The API key is read
    from an environment variable and never stored. `settings`: extra request fields that change the
    vectors (for example a task type); they are stored in the metadata and checked at load time."""

    META_KEYS = ("provider", "model", "dims", "dtype", "instruction", "settings")

    def __init__(self, provider, model, dims, base_url, key_env="EMBEDDING_API_KEY", settings=None,
                 send_dims=True, retries=5, timeout=60):
        import os
        self.key = os.environ.get(key_env)
        if not self.key:
            raise RuntimeError(f"Set the API key in ${key_env}.")
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model, self.dims, self.send_dims = model, int(dims), send_dims
        self.settings = settings or {}
        self.retries, self.timeout = retries, timeout
        self.meta = {"provider": provider, "model": model, "dims": str(self.dims), "dtype": "float32",
                     "instruction": "", "settings": json.dumps(self.settings, sort_keys=True)}

    def __call__(self, texts):
        import urllib.error
        import urllib.request
        body = {"model": self.model, "input": list(texts), "encoding_format": "float", **self.settings}
        if self.send_dims:
            body["dimensions"] = self.dims
        req = urllib.request.Request(self.url, json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())["data"]
                break
            except urllib.error.HTTPError as e:  # retry rate limits and server errors only
                if e.code != 429 and e.code < 500 or attempt == self.retries - 1:
                    raise RuntimeError(f"{self.url}: HTTP {e.code}: {e.read()[:500]!r}") from None
            except urllib.error.URLError:
                if attempt == self.retries - 1:
                    raise
            time.sleep(2 ** attempt)
        vecs = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32)
        assert vecs.shape == (len(texts), self.dims), f"got {vecs.shape}, expected ({len(texts)}, {self.dims})"
        return vecs


def build_model(d_in, mu, sigma, residual=True, hidden=2048, dropout=0.1):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Eddy(nn.Module):
        """y = (W x̂ + b + MLP(x̂)) * sigma + mu, with x̂ = x / |x|. The MLP's last layer starts at zero."""

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d_in, 4096)
            self.residual = residual
            if residual:
                self.mlp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(hidden, 4096))
                nn.init.zeros_(self.mlp[-1].weight)
                nn.init.zeros_(self.mlp[-1].bias)
            self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
            self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))

        def standardized(self, x):
            x = F.normalize(x, dim=-1)
            return self.linear(x) + (self.mlp(x) if self.residual else 0)

        def forward(self, x):
            return self.standardized(x) * self.sigma + self.mu

    return Eddy()


def save_adapter(model, path, meta):
    from safetensors.torch import save_file
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path), metadata={k: str(v) for k, v in meta.items()})


def load_adapter(path, device="cpu"):
    """Load an adapter. The metadata records what it was fit on (model, dims, provider, ...); it is not checked:
    giving the adapter vectors of the right model is the caller's job."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    state = load_file(str(path))
    model = build_model(int(meta["d_in"]), state["mu"].numpy(), state["sigma"].numpy(),
                        residual=meta["residual"] == "True", hidden=int(meta.get("hidden", 2048)))
    model.load_state_dict(state)
    return model.to(device).eval(), meta



class EddyTextEncoder:
    """Same interface as ARDY's LLM2VecEncoder: __call__(texts) -> ([B, 1, 4096], lengths).
    `embedder(texts) -> [B, d_in]` is any callable, from any host; it must serve the model the adapter was fit on
    (`self.meta["model"]`). Only the vector size is checked."""

    def __init__(self, embedder, adapter_path, device="cuda", dtype="float32", cache_size=4096):
        import torch
        self.embedder = embedder
        self.model, self.meta = load_adapter(adapter_path, device)
        self.device, self.dtype = device, getattr(torch, dtype)
        self.cache, self.cache_size = OrderedDict(), cache_size

    def __call__(self, texts):
        import torch
        out = torch.zeros(len(texts), 1, 4096, dtype=self.dtype, device=self.device)
        todo = [t for t in dict.fromkeys(texts) if t.strip() and t not in self.cache]
        if todo:  # empty prompts are never sent: classifier-free guidance needs an exact zero
            x = torch.as_tensor(self.embedder(todo), dtype=torch.float32, device=self.device)
            if x.shape[-1] != self.model.linear.in_features:
                raise ValueError(f"the embedder gives {x.shape[-1]}-d vectors; this adapter was fit on "
                                 f"{self.meta.get('model')} with {self.model.linear.in_features} dimensions")
            with torch.no_grad():
                y = self.model(x)
            for t, v in zip(todo, y):
                self.cache[t] = v.to(self.dtype)
        for i, t in enumerate(texts):
            if t.strip():
                self.cache.move_to_end(t)
                out[i, 0] = self.cache[t]
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return out, [1] * len(texts)

    def to(self, device):
        self.model.to(device)
        self.device = device
        self.cache.clear()
        return self

    def eval(self):
        self.model.eval()
        return self

