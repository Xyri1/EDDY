"""EDDY: map a provider text embedding to the LLM2Vec vector that ARDY expects.

Data, training and evaluation commands (inference code: eddy_encoder.py). See docs/steps/ for the plan.

  prepare        captions -> <dir>/captions.jsonl (HumanML3D or BONES-SEED; dedupe, grouped split,
                 minimal pairs, rewordings). CPU.
  reword-sources pick SEED validation labels to reword (for ε) -> JSONL template. CPU.
  embed          one encoder over all captions -> <dir>/emb/<name>/. GPU.
  pilot          baselines (mean, Procrustes, ridge) and ε from rewordings, on validation. CPU.
  train          EDDY (linear or linear + MLP), best checkpoint by validation e_c -> adapter. GPU or CPU.

pilot and train read one or more dataset dirs (--data data/full data/seed), each with captions.jsonl
and emb/. Train splits are pooled; every dataset is reported on its own; --select picks the dataset
whose validation makes the choices (default: the last one).

Rules (AGENTS.md): the test split is read only with --test. mu and sigma come from train only.
Similarity for R@1 and minimal pairs is cosine after subtracting the train mean mu.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# The inference part lives in eddy_encoder.py (no other dependencies); re-exported here.
from eddy_encoder import (APIEmbedder, EddyTextEncoder, QwenEmbedder, build_model, l2n,  # noqa: F401
                          load_adapter, save_adapter)

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SHARD = 512
# Word swaps for minimal pairs. Each pair differs in one detail only.
SWAPS = [("left", "right"), ("forward", "backward"), ("forwards", "backwards"),
         ("slowly", "quickly"), ("clockwise", "counterclockwise")]


# ---------------------------------------------------------------- prepare

def _split_of(group: str, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) % 10_000 / 10_000
    acc = 0.0
    for name, frac in SPLITS.items():
        acc += frac
        if h < acc:
            return name
    return "test"


def _swap(text: str):
    """Return the text with exactly one swap word replaced, or None."""
    for a, b in SWAPS:
        for x, y in ((a, b), (b, a)):
            pat = re.compile(rf"\b{x}\b", re.I)
            if len(pat.findall(text)) == 1 and not re.search(rf"\b{y}\b", text, re.I):
                return pat.sub(lambda m: y if m.group(0).islower() else y.capitalize(), text)
    return None


def seed_content(filename: str) -> str:
    """BONES-SEED content name from a motion filename: drop actor, mirror suffix, and take number.
    'Idle_Left_001__A017_M' -> 'Idle_Left'. Mirrored motions (_M) stay in the group of their source."""
    return re.sub(r"_\d+$", "", re.sub(r"__A\d+(_M)?$", "", filename))


def _seed_motions(timelines, splits_dir, seed, max_motions=None):
    """(motion, split) for BONES-SEED, held-out motions first so that a text shared with a train
    motion lands in the held-out split (dedupe keeps the first copy).
    test:     every content group with a motion in the benchmark's content test split (whole group).
    test_rep: benchmark repetition-test motions. Their content is also in train, so they are not a
              clean test set; they are only held out (never trained on).
    val / train: the other groups, by group hash (about 10 % val)."""
    split_of = {}
    for name in ("train", "test_content", "test_repetition"):
        for line in (Path(splits_dir) / f"{name}_split_paths.txt").read_text(encoding="utf8").splitlines():
            if line.strip():  # lines, not words: some filenames contain spaces
                split_of[line.strip().split("/")[-1]] = name
    motions = [json.loads(l) for l in Path(timelines).read_text(encoding="utf8").splitlines() if l.strip()]
    unknown = [m["filename"] for m in motions if m["filename"] not in split_of]
    if unknown:
        raise ValueError(f"{len(unknown)} motions are in no benchmark split, for example {unknown[:3]}")
    random.Random(seed).shuffle(motions)
    if max_motions:  # smoke test only
        motions = motions[:max_motions]
    test_groups = {seed_content(m["filename"]) for m in motions if split_of[m["filename"]] == "test_content"}
    out = []
    for m in motions:
        g = seed_content(m["filename"])
        if g in test_groups:
            out.append((0, m, "test"))
        elif split_of[m["filename"]] == "test_repetition":
            out.append((1, m, "test_rep"))
        else:
            out.append((2, m, "val" if _split_of(g, seed) == "val" else "train"))
    out.sort(key=lambda t: t[0])  # stable: keeps the shuffled order within each priority
    return [(m, s) for _, m, s in out]


def prepare(texts_dir, extra, out, seed, max_captions=None, ardy_prompts=None,
            seed_timelines=None, seed_splits=None, seed_max_motions=None, reword=None):
    rows, seen = [], {}  # normalized text -> row id
    norm = lambda t: " ".join(t.lower().split())

    def add(text, group, split=None, pair=None, **extra_fields):
        key = norm(text)
        if not key or key in seen:
            return False
        seen[key] = len(rows)
        rows.append({"id": len(rows), "text": text.strip(), "group": group,
                     "split": split or _split_of(group, seed), "pair": pair, **extra_fields})
        return True

    # Eval-only prompts first: a HumanML3D caption with the same text is then dropped from train (no leak).
    if ardy_prompts:  # fixed prompts for the ε calibration in ARDY (step 4)
        for i, line in enumerate(Path(ardy_prompts).read_text(encoding="utf8").splitlines()):
            add(line, f"ardy:{i}", split="ardy")
    for path in extra or []:  # stream-style prompts: one per line, evaluated apart, never trained on
        for i, line in enumerate(Path(path).read_text(encoding="utf8").splitlines()):
            add(line, f"stream:{Path(path).stem}:{i}", split="stream")
    files = sorted(Path(texts_dir).glob("*.txt")) if texts_dir else []
    random.Random(seed).shuffle(files)
    for f in files:
        # HumanML3D: "M012345" is the left/right mirror of "012345". Same group, or the split leaks.
        group = f.stem.lstrip("M")
        for line in f.read_text(errors="replace").splitlines():
            add(line.split("#")[0], group)
        if max_captions and len(rows) >= max_captions:
            break
    if seed_timelines:  # BONES-SEED labels: the text style ARDY was trained on (docs/steps/02-data.md)
        for m, split in _seed_motions(seed_timelines, seed_splits, seed, seed_max_motions):
            g = seed_content(m["filename"])
            # A few labels are leaked annotation template ("... overview description ... Event 1 ...").
            ok = lambda t: "description" not in t.lower()
            if ok(m["overview_description"]):
                add(m["overview_description"], g, split)
            for e in m["events"]:  # duration: the motion check in step 4 generates this long
                if ok(e["description"]):
                    add(e["description"], g, split, duration=round(e["end_time"] - e["start_time"], 3))

    # Minimal pairs from held-out captions only: (original, swapped). Never trained on.
    n_pairs = 0
    for r in [r for r in rows if r["split"] in ("val", "test")]:
        other = _swap(r["text"])
        if not other or r["pair"] is not None:
            continue
        j = seen.get(" ".join(other.lower().split()))
        if j is None:  # new text: add it as an eval-only row
            add(other, r["group"], split=f"pairs_{r['split']}", pair=r["id"])
            j = rows[-1]["id"]
        elif rows[j]["split"] != r["split"] or rows[j]["pair"] is not None:
            continue  # exists elsewhere (for example in train): not a clean held-out pair
        r["pair"], rows[j]["pair"] = j, r["id"]
        n_pairs += 1

    # Rewordings of validation labels, for ε (step 4). Added last, so earlier row ids do not change.
    n_dup = 0
    for line in (Path(reword).read_text(encoding="utf8").splitlines() if reword else []):
        if not line.strip():
            continue
        item = json.loads(line)
        j = seen.get(norm(item["source"]))
        if j is None or rows[j]["split"] != "val":
            raise ValueError(f"reword source is not a validation caption: {item['source']!r}")
        for t in item["rewordings"]:
            n_dup += not add(t, rows[j]["group"], split="reword", source=j)
    if n_dup:
        print(f"{n_dup} rewordings dropped: same text as an existing caption")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = {s: sum(r["split"] == s for r in rows) for s in sorted({r["split"] for r in rows})}
    print(f"{len(rows)} captions -> {out}: {counts}, {n_pairs} minimal pairs")


def load_captions(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf8").splitlines() if l.strip()]


def reword_sources(captions, n, out, seed=0):
    """Pick n validation event labels (one per content group) and write the rewording template:
    one JSON line per label, {"source": text, "rewordings": []}. An LLM fills in "rewordings"."""
    rows = [r for r in load_captions(captions) if r["split"] == "val" and r.get("duration")]
    random.Random(seed).shuffle(rows)
    picked, groups = [], set()
    for r in rows:
        if r["group"] not in groups:
            groups.add(r["group"])
            picked.append(r)
        if len(picked) == n:
            break
    if len(picked) < n:
        print(f"WARNING only {len(picked)} validation groups, wanted {n}")
    Path(out).write_text("".join(json.dumps({"source": r["text"], "rewordings": []}) + "\n" for r in picked))
    print(f"{len(picked)} sources -> {out}")


# Subject of a SEED label ("A person", "The injured man", "Someone"), dropped in clause joins (", then ...").
_SUBJECT = re.compile(r"^(?:(?:a|an|the)\s+(?:[\w-]+\s+){0,2}?(?:person|individual|man|woman|dancer|figure|"
                      r"character|performer|actor)|someone)\s+(?=(?!is\b)[a-z]*[^s\W]s\b)", re.I)  # a verb follows: "walks"


def _join(texts, rng):
    """Join consecutive event labels into one multi-event text, in the styles of the benchmark's
    timeline-multi texts: "A. Then, B.", "A. Next, the person B.", "A, then B." The order is kept."""
    out = texts[0].rstrip()
    for t in texts[1:]:
        t = t.strip()
        rest = _SUBJECT.sub("", t, count=1)
        style = rng.randrange(5) if rest != t else rng.randrange(2)
        if style < 2:  # new sentence, own subject: "Then, a person ..."
            out = f"{out} {('Then', 'Next', 'After that')[rng.randrange(3)]}, {t[0].lower()}{t[1:]}"
        elif style == 2:  # new sentence, "the person" ("they" would need the verb changed: "they walk")
            out = f"{out} {('Then', 'Next', 'After that')[rng.randrange(3)]}, the person {rest}"
        else:  # one sentence: "A, then B." / "A and then B."
            out = f"{out.rstrip('.')}{(', then', ' and then')[style - 3]} {rest}"
    return out if out.endswith((".", "!", "?")) else out + "."


def seed_joins(timelines, splits_dir, out, n_train=20000, n_val=2000, exclude=(), seed=0):
    """Joined multi-event texts from BONES-SEED (step 3, the timeline-multi fix). One text per motion:
    2 (70 %) or 3 (30 %) consecutive event labels, joined by _join. Split and group come from
    _seed_motions, so they match data/seed; test and test_rep motions are never used. Texts already
    in an `exclude` dataset are skipped. duration: from the first event's start to the last one's end."""
    rng = random.Random(seed)
    norm = lambda t: " ".join(t.lower().split())
    seen = {norm(r["text"]) for d in exclude for r in load_captions(Path(d) / "captions.jsonl")}
    want = {"train": n_train, "val": n_val}
    rows = []
    for m, split in _seed_motions(timelines, splits_dir, seed):
        ev = [e for e in m["events"] if "description" not in e["description"].lower()]
        if split not in want or len(ev) < 2 or sum(r["split"] == split for r in rows) >= want[split]:
            continue
        k = min(len(ev), 3 if rng.random() < 0.3 else 2)
        i = rng.randrange(len(ev) - k + 1)
        win = ev[i: i + k]
        text = _join([e["description"] for e in win], rng)
        if norm(text) in seen:
            continue
        seen.add(norm(text))
        rows.append({"id": len(rows), "text": text, "group": seed_content(m["filename"]), "split": split,
                     "pair": None, "events": k, "duration": round(win[-1]["end_time"] - win[0]["start_time"], 3)})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = {s: sum(r["split"] == s for r in rows) for s in want}
    print(f"{len(rows)} joined texts -> {out}: {counts}")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_llm2vec(device="cuda", repo=None):
    """ARDY's own LLM2Vec encoder, so targets come from the exact runtime code path.
    `repo`: a premerged quantized checkpoint (for example voxta/Llama-3-8B-LLM2Vec-ARDY-INT8),
    loaded the same way as hardy/experiments/02-quantized-encoder."""
    import os
    os.environ["TEXT_ENCODER_DEVICE"] = device
    if repo is None:
        from ardy.model.load_model import load_text_encoder
        return load_text_encoder(mode="local", device=device)
    from ardy.model.llm2vec import LLM2VecEncoder
    from huggingface_hub import snapshot_download
    enc = LLM2VecEncoder(base_model_name_or_path=snapshot_download(repo), peft_model_name_or_path=None,
                         dtype="bfloat16", llm_dim=4096, device=device)
    enc.model.model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # keeps Llama-3 prompt format
    assert enc.model.model.is_quantized
    return enc


def embed(captions, out_dir, encoder, model_path=None, repo=None, dims=None, batch=64, device="cuda",
          api=None):
    """Embed every caption (all splits) into shards. Skips shards that exist, so it can resume."""
    import importlib.metadata as md
    try:
        import torch
    except ImportError:  # the API path needs no torch
        torch = None
    rows = load_captions(captions)
    texts = [r["text"] for r in rows]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp, sha = out / "captions.sha256", file_sha256(captions)
    if stamp.exists() and stamp.read_text(encoding="utf8") != sha:  # resume only shards built from the same captions
        raise RuntimeError(f"{out} holds shards from other captions. Use a new --out or delete it.")
    stamp.write_text(sha)
    if encoder == "llm2vec":
        enc = load_llm2vec(device, repo)
        feats_dtype = []

        def fn(chunk):
            vecs = []
            for t in chunk:  # one prompt per call, exactly like ARDY at runtime (no padding effects)
                features, lengths = enc([t])
                assert tuple(features.shape) == (1, 1, 4096) and list(lengths) == [1]
                feats_dtype.append(str(features.dtype))
                vecs.append(features[0, 0].float().cpu().numpy())
            return np.stack(vecs)
        meta = {"encoder": "llm2vec", "repo": repo or "bf16 (ARDY load_text_encoder)", "dims": 4096}
    elif encoder == "api":
        enc = APIEmbedder(**api)
        fn, feats_dtype = enc, None
        meta = {"encoder": "api", "base_url": api["base_url"], **enc.meta}
    else:
        enc = QwenEmbedder(model_path, dims, device)
        fn = enc
        feats_dtype = None
        meta = {"encoder": "qwen", **enc.meta}
    step = 1 if encoder == "llm2vec" else batch
    t0 = time.time()
    for s in range(0, len(texts), SHARD):
        part = out / f"part_{s // SHARD:05d}.npy"
        if part.exists():
            continue
        chunk = texts[s: s + SHARD]
        vecs = np.concatenate([fn(chunk[i: i + step]) for i in range(0, len(chunk), step)])
        assert np.isfinite(vecs).all()
        np.save(part, vecs.astype(np.float32))
        print(f"{min(s + SHARD, len(texts))}/{len(texts)}  {time.time() - t0:.0f} s", flush=True)
    meta.update({
        "n": len(texts), "captions_sha256": file_sha256(captions),
        "features_dtype": feats_dtype[0] if feats_dtype else "float32",
        "gpu": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "cpu",
        "versions": {p: md.version(p) for p in ("torch", "transformers") if torch},
        "date": time.strftime("%Y-%m-%d"),  # API models can change without notice
    })
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done -> {out}")


def load_emb(path, captions_path=None):
    path = Path(path)
    parts = sorted(path.glob("part_*.npy"))
    x = np.concatenate([np.load(p) for p in parts]).astype(np.float64)
    meta = json.loads((path / "meta.json").read_text(encoding="utf8"))
    assert len(x) == meta["n"], f"{path}: {len(x)} rows, meta says {meta['n']}. Embedding is incomplete."
    if captions_path:
        assert meta["captions_sha256"] == file_sha256(captions_path), f"{path} was built from other captions."
    return x, meta


# ---------------------------------------------------------------- ARDY embed_text

def load_embed_text(sources):
    """All `embed_text` weights (root and body block) from ARDY denoiser checkpoints.
    A source is a denoiser.safetensors path or an ARDY repo ID in the HF cache."""
    from safetensors.torch import load_file
    out = {}
    for src in sources:
        p = Path(src)
        if not p.exists():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(src)) / "denoiser.safetensors"
        state = load_file(str(p))
        keys = [k for k in state if k.endswith("embed_text.weight")]
        assert len(keys) == 2, f"{p}: expected 2 embed_text weights (root, body), found {keys}"
        for k in keys:
            w = state[k].float().numpy().astype(np.float64)
            assert w.shape[1] == 4096, f"{k}: {w.shape}"
            out[f"{src}:{k}"] = w
            print(f"embed_text {src} {k}: {tuple(w.shape)} (latent_dim = {w.shape[0]})")
    return out


# ---------------------------------------------------------------- baselines (numpy)

def fit_ridge(x, z, lam):
    """Closed-form ridge from L2-normalized x to standardized z. Returns (A [d_in, d_out], a [d_out])."""
    x = l2n(x)
    xm, zm = x.mean(0), z.mean(0)
    xc = x - xm
    A = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ (z - zm))
    return A, zm - xm @ A


def fit_procrustes(x, z):
    """Orthogonal (semi-orthogonal if d_in != d_out) map plus one global scale. No other scaling."""
    x = l2n(x)
    xm, zm = x.mean(0), z.mean(0)
    xc, zc = x - xm, z - zm
    u, s, vt = np.linalg.svd(xc.T @ zc, full_matrices=False)
    R = u @ vt
    A = R * (s.sum() / (xc ** 2).sum())
    return A, zm - xm @ A


def apply_linear(x, A, a):
    return l2n(x) @ A + a


# ---------------------------------------------------------------- metrics

def centered_unit(y, mu):
    return l2n(y - mu)


def retrieval(pred, target, mu):
    """R@1, R@5 and the largest hub (how many queries share one top-1 target). Gallery = target rows."""
    s = centered_unit(pred, mu) @ centered_unit(target, mu).T
    true = np.diag(s)
    rank = (s >= true[:, None]).sum(1) - 1  # ties count against the query (a constant prediction scores 0)
    top1 = s.argmax(1)
    return {"R@1": float((rank == 0).mean()), "R@5": float((rank < 5).mean()),
            "hub_max": int(np.bincount(top1, minlength=len(s)).max()), "gallery": len(s)}


def pair_accuracy(pred_a, pred_b, y_a, y_b, mu):
    pa, pb = centered_unit(pred_a, mu), centered_unit(pred_b, mu)
    ya, yb = centered_unit(y_a, mu), centered_unit(y_b, mu)
    ok = np.concatenate([(pa * ya).sum(1) > (pa * yb).sum(1), (pb * yb).sum(1) > (pb * ya).sum(1)])
    return float(ok.mean())


def ardy_error(pred, y, mu, Ws):
    """ARDY-space error per embed_text weight W.
    e   = |W(pred - y)| / |W y|        (as in docs; the mean offset W mu inflates the denominator)
    e_c = |W(pred - y)| / |W (y - mu)|  (relative to the part that varies between prompts)"""
    out = {}
    for name, W in Ws.items():
        num = np.linalg.norm((pred - y) @ W.T, axis=1)
        e = num / np.linalg.norm(y @ W.T, axis=1)
        ec = num / np.linalg.norm((y - mu) @ W.T, axis=1)
        out[name] = {"e_median": float(np.median(e)), "e_max": float(e.max()),
                     "e_c_median": float(np.median(ec)), "e_c_p95": float(np.percentile(ec, 95)),
                     "e_c_max": float(ec.max())}
    return out


def worst(ardy, key="e_c_median"):
    """The worst embed_text weight for one statistic: max over weights."""
    return max(v[key] for v in ardy.values())


def effective_rank(yc, fracs=(0.9, 0.95, 0.99)):
    s = np.linalg.svd(yc, compute_uv=False) ** 2
    cum = np.cumsum(s) / s.sum()
    return {f"{int(f * 100)}%": int(np.searchsorted(cum, f) + 1) for f in fracs}


# ---------------------------------------------------------------- data views

class View:
    """One dataset's held-out rows, its minimal pairs, and its stream prompts (global row indices)."""

    def __init__(self, held, pairs, stream):
        self.held, self.pairs, self.stream = held, pairs, stream


class Data:
    """One or more dataset dirs (each: captions.jsonl, emb/). Rows are concatenated in --data order.
    Train is pooled over all dirs; held-out views are per dir."""

    def __init__(self, dirs, use_test=False):
        self.dirs = [Path(p) for p in dirs]
        self.rows = []
        for p in self.dirs:
            off = len(self.rows)
            for i, r in enumerate(load_captions(p / "captions.jsonl")):
                assert r["id"] == i, f"{p}: row ids must be 0..n-1 in order"
                r = {**r, "id": i + off, "src": p.name}
                for k in ("pair", "source"):
                    if r.get(k) is not None:
                        r[k] += off
                self.rows.append(r)
        self.names = [p.name for p in self.dirs]
        assert len(set(self.names)) == len(self.names), f"dataset dir names must differ: {self.names}"
        self.train = self.idx("train")
        self.held_name = "test" if use_test else "val"
        rw = [(r["id"], r["source"]) for r in self.rows if r["split"] == "reword"]
        self.reword = np.array(rw, dtype=int).reshape(-1, 2)  # (rewording, source label)

    def idx(self, split, src=None):
        return np.array([r["id"] for r in self.rows if r["split"] == split and src in (None, r["src"])], dtype=int)

    def view(self, src, held=None):
        held = held or self.held_name
        pairs = {tuple(sorted((r["id"], r["pair"]))) for r in self.rows
                 if r["src"] == src and r["pair"] is not None and r["split"] in (held, f"pairs_{held}")}
        return View(self.idx(held, src), np.array(sorted(pairs), dtype=int).reshape(-1, 2),
                    self.idx("stream", src))

    def load(self, name):
        """Embedding folder `name` from every dir, concatenated in row order."""
        parts = [load_emb(p / "emb" / name, p / "captions.jsonl") for p in self.dirs]
        return np.concatenate([x for x, _ in parts]), parts[0][1]


def evaluate(pred_fn, x, y, mu, v, Ws):
    """All metrics on one view: held-out split (val unless --test), its minimal pairs, stream prompts."""
    res = retrieval(pred_fn(x[v.held]), y[v.held], mu)
    if len(v.pairs):
        res["pairs_acc"] = pair_accuracy(pred_fn(x[v.pairs[:, 0]]), pred_fn(x[v.pairs[:, 1]]),
                                         y[v.pairs[:, 0]], y[v.pairs[:, 1]], mu)
        res["n_pairs"] = len(v.pairs)
    if len(v.stream):
        res["stream_R@1"] = retrieval(pred_fn(x[v.stream]), y[v.stream], mu)["R@1"]
    if Ws:
        res["ardy"] = ardy_error(pred_fn(x[v.held]), y[v.held], mu, Ws)
    return res


def summary_row(name, r):
    a = r.get("ardy")
    stat = lambda k: f"{worst(a, k):.4f}" if a else "nan"
    cells = [name, f"{r['R@1']:.3f}", f"{r['R@5']:.3f}", f"{r['hub_max']}/{r['gallery']}",
             f"{r.get('pairs_acc', float('nan')):.3f}", f"{r.get('stream_R@1', float('nan')):.3f}",
             stat("e_median"), stat("e_c_median"), stat("e_c_p95"), stat("e_c_max")]
    return "| " + " | ".join(cells) + " |"


HEADER = ("| model [data] | R@1 | R@5 | hub max / gallery | pairs acc | stream R@1 | e median | e_c median "
          "| e_c p95 | e_c max |\n|---|---|---|---|---|---|---|---|---|---|")


# ---------------------------------------------------------------- pilot

def epsilon(y, mu, d, Ws):
    """ε from rewordings (step 4): e_c of each rewording's LLM2Vec vector against its source label's.
    ε = median, ε₉₅ = 95th percentile; worst embed_text weight for each."""
    if not (len(d.reword) and Ws):
        return None
    a = ardy_error(y[d.reword[:, 0]], y[d.reword[:, 1]], mu, Ws)
    return {"eps": worst(a, "e_c_median"), "eps95": worst(a, "e_c_p95"), "n_rewordings": len(d.reword),
            "n_sources": len(set(d.reword[:, 1].tolist()))}


def ridge_curve(x, z, y, mu, sigma, d, views, Ws, lams, sel_val, sizes, seed=0):
    """Ridge on random train subsets (step 3 learning curve). lambda is chosen again for each size, on
    --select val. Returns {size: {dataset: worst-weight e_c median / p95 / max}}."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in sorted(sizes):
        tr = np.sort(rng.choice(d.train, n, replace=False)) if n < len(d.train) else d.train
        best = None
        for lam in lams:
            A, a = fit_ridge(x[tr], z[tr], lam)
            mse = float(((apply_linear(x[sel_val], A, a) - z[sel_val]) ** 2).mean())
            if best is None or mse < best[0]:
                best = (mse, lam, A, a)
        _, lam, A, a = best
        row = {"n": len(tr), "lambda": lam}
        for s, v in views.items():
            if len(v.held) and Ws:
                e = ardy_error(apply_linear(x[v.held], A, a) * sigma + mu, y[v.held], mu, Ws)
                row[s] = {k: worst(e, k) for k in ("e_c_median", "e_c_p95", "e_c_max")}
        out[len(tr)] = row
        print(f"curve n={len(tr)} lambda={lam}: " + ", ".join(
            f"{s} e_c median {r['e_c_median']:.4f}" for s, r in row.items() if isinstance(r, dict)), flush=True)
    return out


def pilot(dirs, target, inputs, compare, ardy_ckpt, lams, out, use_test=False, select=None, curve=None):
    d = Data(dirs, use_test)
    select = select or d.names[-1]
    y, _ = d.load(target)
    mu, sigma = y[d.train].mean(0), y[d.train].std(0) + 1e-6
    z = (y - mu) / sigma
    Ws = load_embed_text(ardy_ckpt) if ardy_ckpt else {}
    views = {src: d.view(src) for src in d.names}
    report = {"split": d.held_name, "select": select, "n_train": len(d.train),
              "n_held": {s: len(v.held) for s, v in views.items()}, "rows": {}}

    # Effective rank of the targets: how many directions carry the variance.
    yc = y[d.train] - mu
    report["effective_rank"] = {"raw": effective_rank(yc)}
    for name, W in Ws.items():
        report["effective_rank"][name] = effective_rank(yc @ W.T)
    # Leak check: held-out captions whose target is nearly identical to a train target.
    ytr = centered_unit(y[d.train], mu)
    report["near_dup_held_vs_train(>0.95)"] = {
        s: int(((centered_unit(y[v.held], mu) @ ytr.T).max(1) > 0.95).sum()) if len(v.held) else 0
        for s, v in views.items()}
    report["epsilon"] = epsilon(y, mu, d, Ws)

    rows = report["rows"]
    sel_val = d.view(select, "val").held  # every choice is made here, even with --test

    def add_rows(name, pred_fn, x, extra=None):
        for s, v in views.items():
            if len(v.held):
                rows[f"{name} [{s}]"] = {**evaluate(pred_fn, x, y, mu, v, Ws), **(extra or {})}

    add_rows("mean (floor)", lambda x: np.repeat(mu[None], len(x), 0), y)
    for name in compare:  # quantized LLM2Vec etc.: already in the target space, no adapter
        x, _ = d.load(name)
        add_rows(f"{name} (no adapter)", lambda v: v, x)
    for name in inputs:
        x, meta = d.load(name)
        A, a = fit_procrustes(x[d.train], z[d.train])
        add_rows(f"{name} procrustes", lambda v, A=A, a=a: apply_linear(v, A, a) * sigma + mu, x)
        # lambda: validation MSE in the standardized space, on the --select dataset. Not R@1: it
        # saturates at 1.0 on small galleries, so every lambda ties.
        best = None
        for lam in lams:
            A, a = fit_ridge(x[d.train], z[d.train], lam)
            mse = float(((apply_linear(x[sel_val], A, a) - z[sel_val]) ** 2).mean())
            if best is None or mse < best[0]:
                best = (mse, lam, A, a)
        if best[1] in (min(lams), max(lams)):
            print(f"WARNING {name}: best lambda {best[1]} is at the edge of the grid; widen --lambdas")
        _, lam, A, a = best
        add_rows(f"{name} ridge", lambda v: apply_linear(v, A, a) * sigma + mu, x, {"lambda": lam})
        if curve:
            report.setdefault("learning_curve", {})[name] = ridge_curve(x, z, y, mu, sigma, d, views, Ws,
                                                                       lams, sel_val, curve)

    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))
    print(f"\nSplit: {d.held_name}. Choices made on: {select} val. Similarity: cosine after subtracting "
          f"train mu. e and e_c: worst embed_text weight for each statistic, over prompts.\n")
    print(HEADER)
    for name, r in rows.items():
        print(summary_row(name, r))
    if out:
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"\nfull report -> {out}")


# ---------------------------------------------------------------- train

def train(dirs, target, input_name, out, residual=True, init="random", lr=1e-3, wd=1e-4,
          batch=512, steps=3000, eval_every=100, dropout=0.1, hidden=2048, seed=0, limit=None,
          device="cpu", ardy_ckpt=None, select=None, ridge_lambda=0.01, ardy_loss=0.0):
    """Keeps the training checkpoint with the lowest validation score on the --select dataset:
    worst-weight median e_c with --ardy-ckpt (the main metric), else standardized MSE.
    ardy_loss > 0 adds the ARDY-space term: |W(pred - y)|² / mean_train |W(y - mu)|², averaged over
    every embed_text W, so it reads like e_c²."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = Data(dirs)  # validation only: test is never read here
    select = select or d.names[-1]
    y, _ = d.load(target)
    x, in_meta = d.load(input_name)
    Ws = load_embed_text(ardy_ckpt) if ardy_ckpt else {}
    val = d.view(select, "val").held
    mu, sigma = y[d.train].mean(0), y[d.train].std(0) + 1e-6
    z = (y - mu) / sigma
    tr = d.train[:limit] if limit else d.train

    model = build_model(x.shape[1], mu, sigma, residual, hidden, dropout).to(device)
    if init == "ridge":
        A, a = fit_ridge(x[tr], z[tr], ridge_lambda)
        with torch.no_grad():
            model.linear.weight.copy_(torch.as_tensor(A.T))
            model.linear.bias.copy_(torch.as_tensor(a))
    T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
    xt, zt, xv = T(x[tr]), T(z[tr]), T(x[val])
    metric = "e_c_median" if Ws else "mse_std"
    if ardy_loss:
        assert Ws, "--ardy-loss needs --ardy-ckpt"
        # sigma folded into W: W(pred - y) = (W * sigma)(standardized pred - z)
        Wt = [T(W * sigma) / np.sqrt((((y[tr] - mu) @ W.T) ** 2).sum(1).mean()) for W in Ws.values()]

    def score():
        """(score, R@1) on validation; lower score is better."""
        model.eval()
        with torch.no_grad():
            p = model(xv).double().cpu().numpy()
        model.train()
        s = worst(ardy_error(p, y[val], mu, Ws)) if Ws else float((((p - mu) / sigma - z[val]) ** 2).mean())
        return s, retrieval(p, y[val], mu)["R@1"]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)  # constant lr: tune a schedule last
    s, r1 = score()
    best = (s, 0, {k: v.clone() for k, v in model.state_dict().items()})
    print(f"step 0: val {metric} {s:.4f}, R@1 {r1:.4f} ({select}, gallery {len(val)})")
    for step in range(1, steps + 1):
        i = torch.randint(0, len(xt), (min(batch, len(xt)),), device=device)
        diff = model.standardized(xt[i]) - zt[i]
        loss = (diff ** 2).mean()
        if ardy_loss:
            loss = loss + ardy_loss * sum(((diff @ W.T) ** 2).sum(1).mean() for W in Wt) / len(Wt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            s, r1 = score()
            print(f"step {step}: train loss {loss.item():.5f}, val {metric} {s:.4f}, R@1 {r1:.4f}", flush=True)
            if s < best[0]:
                best = (s, step, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    meta = {**{k: in_meta[k] for k in APIEmbedder.META_KEYS if k in in_meta},
            "d_in": x.shape[1], "residual": residual, "hidden": hidden, "seed": seed, "init": init, "ardy_loss": ardy_loss,
            "best_step": best[1], "val_metric": metric, "val_score": round(best[0], 6), "select": select,
            "target": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "captions_sha256": json.dumps({p.name: file_sha256(p / "captions.jsonl") for p in d.dirs})}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_adapter(model, out, meta)
    print(f"best val {metric} {best[0]:.4f} at step {best[1]} -> {out}")
    model.eval()
    pred = lambda v: model(T(v)).double().cpu().numpy()
    with torch.no_grad():  # every dataset on its own row (step 3), same table as pilot
        print("\n" + HEADER)
        for src in d.names:
            v = d.view(src, "val")
            if len(v.held):
                print(summary_row(f"eddy [{src}]", evaluate(pred, x, y, mu, v, Ws)))


# ---------------------------------------------------------------- motion check (docs/steps/04-evaluate.md)

def _sanitize_invariant(t):
    """Conservative test that Kimodo's sanitize_text leaves t unchanged: one sentence, capitalized,
    single spaces, ends with '.'. motion-score checks again with the real function."""
    return (t == " ".join(t.split()) and t[:1].isalpha() and t[0].isupper() and t[1:] == t[1:].lower()
            and t.endswith(".") and not any(c in t[:-1] for c in ".!?"))


def motion_prompts(dirs, out, n=500, input_name="qwen3-8b_4096", target="llm2vec_bf16", adapters=(),
                   ridge_lambda=0.01, seed=0, split="val"):
    """Pick n SEED event texts of `split` from the last dir (one per group, with duration), write one meta.json
    folder per prompt for ARDY's generate_eval.py, and the 4096-d vector of each prompt for every text
    encoder: llm2vec (the stored target), ridge (fit on the pooled train splits, as in step 3), and each
    adapter."""
    out = Path(out)
    d = Data(dirs)
    rows = [r for r in d.rows if r["src"] == d.names[-1] and r["split"] == split and r.get("duration")]
    if any(r.get("events") for r in rows):  # joined texts (data/join) have several sentences: the real function
        sanitize = _kimodo_sanitize()
        rows = [r for r in rows if sanitize(r["text"]) == r["text"]]
    else:  # SEED labels: the conservative check, so the prompt choice stays as in step 4
        rows = [r for r in rows if _sanitize_invariant(r["text"])]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    # round robin over groups: every group once before any group twice (SEED val has 385 groups)
    seen = defaultdict(int)
    order = []
    for k, r in enumerate(rows):
        order.append((seen[r["group"]], k))
        seen[r["group"]] += 1
    assert len(rows) >= n, f"only {len(rows)} rows qualify"
    picked = sorted((rows[k] for _, k in sorted(order)[:n]), key=lambda r: r["id"])
    ids = np.array([r["id"] for r in picked])
    for i, r in enumerate(picked):
        p = out / f"{i:04d}"
        p.mkdir(parents=True, exist_ok=True)
        (p / "meta.json").write_text(json.dumps({"text": r["text"], "duration": r["duration"], "row_id": r["id"]}))
    (out / "texts.json").write_text(json.dumps([r["text"] for r in picked], indent=0))

    y, _ = d.load(target)
    x, _ = d.load(input_name)
    vecs = {"llm2vec": y[ids]}
    if ridge_lambda:
        mu, sigma = y[d.train].mean(0), y[d.train].std(0) + 1e-6
        A, a = fit_ridge(x[d.train], (y[d.train] - mu) / sigma, ridge_lambda)
        vecs["ridge"] = apply_linear(x[ids], A, a) * sigma + mu
    for spec in adapters:  # name=path
        name, path = spec.split("=", 1)
        import torch
        model, _ = load_adapter(path)
        with torch.no_grad():
            vecs[name] = model(torch.as_tensor(x[ids], dtype=torch.float32)).double().numpy()
    (out / "vectors").mkdir(exist_ok=True)
    for name, v in vecs.items():
        np.save(out / "vectors" / f"{name}.npy", v.astype(np.float32))
        print(f"{name}: {v.shape}")
    durs = [r["duration"] for r in picked]
    print(f"{n} prompts -> {out} (duration {min(durs):.1f}-{max(durs):.1f} s, median {np.median(durs):.1f})")


class StoredTextEncoder:
    """ARDY / TMR text-encoder interface backed by stored vectors: __call__(texts) -> ([B, 1, 4096], lengths).
    Empty prompts give an exact zero vector. An unknown text is an error, never a silent fallback."""

    def __init__(self, texts, vectors, device="cuda"):
        assert len(texts) == len(vectors)
        self.table = {t: v for t, v in zip(texts, vectors)}
        self.device = device

    def __call__(self, texts):
        import torch
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), 1, 4096), dtype=np.float32)
        for i, t in enumerate(texts):
            if t.strip():
                out[i, 0] = self.table[t]
        return torch.from_numpy(out).to(self.device), [1] * len(texts)

    def to(self, device=None, dtype=None):
        self.device = device or self.device
        return self

    def eval(self):
        return self


def motion_generate(prompts, vectors, seed, out, model="soma", ardy_repo="ext/ardy", batch=1):
    """Run ARDY's scripts/generate_eval.py over the prompt folders, with stored vectors as text encoder.
    Same prompts, batches and seed give the same noise, so runs differ only in the text vectors.
    Batch 1: a batch that mixes a motion over ~10 s with short ones crashes ARDY's autoregressive
    decoder (a short motion gets 0 valid frames in a late window, and compute_vel_angle indexes -1)."""
    import functools
    import importlib.util
    import torch
    from ardy.model import load_model
    prompts = Path(prompts)
    texts = json.loads((prompts / "texts.json").read_text(encoding="utf8"))
    enc = StoredTextEncoder(texts, np.load(prompts / "vectors" / f"{vectors}.npy"),
                            "cuda" if torch.cuda.is_available() else "cpu")
    spec = importlib.util.spec_from_file_location("generate_eval", Path(ardy_repo) / "scripts" / "generate_eval.py")
    ge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ge)
    ge.load_model = functools.partial(load_model, text_encoder=enc)
    # num_workers 0: workers on Windows re-import modules, and generate_eval is loaded from a file path
    sys.argv = ["generate_eval.py", "--benchmark", str(prompts), "--output", str(out), "--model", model,
                "--seed", str(seed), "--batch_size", str(batch), "--num_workers", "0"]
    ge.main()
    (Path(out) / "run.json").write_text(json.dumps({"vectors": vectors, "seed": seed, "model": model}))


def load_tmr(name="tmr-soma-rp", device="cuda"):
    """TMR without its LLM2Vec. Kimodo's load_model(text_encoder=...) would overwrite TMR's top text
    encoder, so build it here; text goes in as stored LLM2Vec vectors through encode_text."""
    from omegaconf import OmegaConf
    from kimodo.model.load_model import _resolve_hf_model_path, instantiate_from_dict
    from kimodo.model.registry import resolve_model_name
    path = _resolve_hf_model_path(resolve_model_name(name, "TMR"))
    cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(path / "config.yaml"),
                                                 {"checkpoint_dir": str(path)}), resolve=True)
    cfg.pop("checkpoint_dir", None)
    cfg["text_encoder"] = None
    return instantiate_from_dict(cfg, overrides={"device": device}).eval()


def motion_score(prompts, runs, reference=None, out=None, device="cuda"):
    """TMR R@3 of each run's motions against the prompt texts (gallery = all prompts), and FID between
    each run's motion embeddings and the reference run's (default: the first run)."""
    import torch
    from kimodo.metrics.retrieval import (calculate_activation_statistics, calculate_frechet_distance,
                                          compute_tmr_retrieval_metrics)
    from kimodo.sanitize import sanitize_text
    prompts = Path(prompts)
    texts = json.loads((prompts / "texts.json").read_text(encoding="utf8"))
    bad = [t for t in texts if sanitize_text(t) != t]
    assert not bad, f"{len(bad)} texts change under sanitize_text, e.g. {bad[:3]}"
    tmr = load_tmr(device=device)
    y = torch.as_tensor(np.load(prompts / "vectors" / "llm2vec.npy"), device=device)
    with torch.inference_mode():
        t_emb = tmr.encode_text({"x": y[:, None], "mask": torch.ones(len(y), 1, dtype=torch.bool, device=device)},
                                unit_vector=True).cpu().numpy()
    emb = {}
    for run in runs:
        run = Path(run)
        m = []
        for i in range(len(texts)):
            pj = np.load(run / f"{i:04d}" / "motion.npz")["posed_joints"]
            with torch.inference_mode():
                m.append(tmr.encode_motion(torch.as_tensor(pj, dtype=torch.float32, device=device),
                                           unit_vector=True).reshape(-1).cpu().numpy())
        emb[run.name] = np.stack(m)
    ref = Path(reference).name if reference else Path(runs[0]).name
    mu_r, cov_r = calculate_activation_statistics(emb[ref])
    report = {}
    print(f"\n{len(texts)} prompts. FID against {ref}.\n\n| run | R@1 | R@3 | R@10 | FID vs {ref} | FID gen-text |\n|---|---|---|---|---|---|")
    for name, e in emb.items():
        r = compute_tmr_retrieval_metrics(e, t_emb)
        fid = float(calculate_frechet_distance(*calculate_activation_statistics(e), mu_r, cov_r))
        report[name] = {"R@1": r["TMR/t2m_R/R01"], "R@3": r["TMR/t2m_R/R03"], "R@10": r["TMR/t2m_R/R10"],
                        "FID_vs_ref": fid, "FID_gen_text": r["TMR/FID/gen_text"]}
        q = report[name]
        print(f"| {name} | {q['R@1']:.2f} | {q['R@3']:.2f} | {q['R@10']:.2f} | {fid:.4f} | {q['FID_gen_text']:.4f} |")
    if out:
        Path(out).write_text(json.dumps({"reference": ref, "n": len(texts), "runs": report}, indent=2))
        print(f"\n-> {out}")


# ---------------------------------------------------------------- Kimodo benchmark (docs/steps/04-evaluate.md#gate)

def _kimodo_sanitize(kimodo_repo="ext/kimodo"):
    """Kimodo's sanitize_text (TMR cleans text before LLM2Vec), loaded from its file: pure Python, no deps."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("kimodo_sanitize", Path(kimodo_repo) / "kimodo" / "sanitize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sanitize_text


def bench_vectors(suite, dirs, adapter, name="eddy", input_name="qwen3-8b_4096", target="llm2vec_bf16",
                  exclude_to=None):
    """Write <suite>/texts.json and <suite>/vectors/{llm2vec,<name>}.npy for every text in the suite's meta.json
    files, plus the sanitized form TMR embeds. Vectors come from the stored data (first row with that text).
    exclude_to: move sample folders whose raw or sanitized text has no stored vectors out of the (built) suite
    into this folder, keeping their relative path, and list them in <suite>/excluded.json."""
    import shutil
    import torch
    suite = Path(suite)
    sanitize = _kimodo_sanitize()
    d = Data(dirs)
    row = {}
    for r in d.rows:
        row.setdefault(r["text"], r["id"])
    if exclude_to:
        excluded = []
        for p in sorted(suite.rglob("meta.json")):
            t = json.loads(p.read_text(encoding="utf8"))["text"]
            if t and (t not in row or sanitize(t) not in row):
                rel = p.parent.relative_to(suite)
                (Path(exclude_to) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p.parent), str(Path(exclude_to) / rel))
                excluded.append({"sample": rel.as_posix(), "text": t})
        (suite / "excluded.json").write_text(json.dumps(excluded, indent=1))
        print(f"excluded {len(excluded)} samples without stored vectors -> {exclude_to}")
    raw = sorted({json.loads(p.read_text(encoding="utf8"))["text"] for p in suite.rglob("meta.json")} - {""})
    texts = sorted(set(raw) | {sanitize(t) for t in raw})
    missing = [t for t in texts if t not in row]
    missing_raw = [t for t in missing if t in raw]
    print(f"{len(raw)} suite texts, {len(texts) - len(raw)} extra sanitized forms; missing from the data: "
          f"{len(missing_raw)} suite texts, {len(missing) - len(missing_raw)} sanitized forms")
    for t in missing[:10]:
        print(f"  missing: {t!r}")
    assert not missing_raw, "every suite text needs stored vectors (ARDY reads the raw text)"
    texts = [t for t in texts if t in row]
    ids = [row[t] for t in texts]
    y, _ = d.load(target)
    x, _ = d.load(input_name)
    model, _ = load_adapter(adapter)
    with torch.no_grad():
        pred = model(torch.as_tensor(x[ids], dtype=torch.float32)).numpy()
    (suite / "vectors").mkdir(exist_ok=True)
    (suite / "texts.json").write_text(json.dumps(texts, indent=0))
    np.save(suite / "vectors" / "llm2vec.npy", y[ids].astype(np.float32))
    np.save(suite / "vectors" / f"{name}.npy", pred.astype(np.float32))
    (suite / "vectors" / "missing_sanitized.json").write_text(json.dumps(missing, indent=0))
    print(f"-> {suite}/texts.json, vectors/llm2vec.npy, vectors/{name}.npy ({len(texts)} texts)")


def bench_embed(run, vectors_dir, device="cuda", overwrite=False):
    """Kimodo's embed_folder.py, with TMR's text side fed by stored LLM2Vec vectors (LLM2Vec is never loaded):
    writes motion_embedding.npy, gt_motion_embedding.npy, text_embedding.npy next to each meta.json in `run`."""
    import torch
    run, vectors_dir = Path(run), Path(vectors_dir)
    sanitize = _kimodo_sanitize()
    texts = json.loads((vectors_dir / "texts.json").read_text(encoding="utf8"))
    y = np.load(vectors_dir / "vectors" / "llm2vec.npy")
    table = dict(zip(texts, y))
    tmr = load_tmr(device=device)
    metas = sorted(p for p in run.rglob("meta.json") if (p.parent / "motion.npz").is_file())
    skipped = []
    for k, meta_path in enumerate(metas):
        sd = meta_path.parent
        if not overwrite and (sd / "text_embedding.npy").is_file():
            continue
        text = sanitize(json.loads(meta_path.read_text(encoding="utf8"))["text"])
        if text not in table:
            skipped.append(text)
            continue
        with torch.inference_mode():
            for src, dst in (("motion.npz", "motion_embedding.npy"), ("gt_motion.npz", "gt_motion_embedding.npy")):
                if (sd / src).is_file():
                    pj = np.load(sd / src)["posed_joints"]
                    pj = pj[0] if pj.ndim == 4 else pj
                    e = tmr.encode_motion(torch.as_tensor(pj, dtype=torch.float32, device=device), unit_vector=True)
                    np.save(sd / dst, e.cpu().numpy())
            v = torch.as_tensor(table[text], device=device)[None, None]
            e = tmr.encode_text({"x": v, "mask": torch.ones(1, 1, dtype=torch.bool, device=device)}, unit_vector=True)
            np.save(sd / "text_embedding.npy", e.cpu().numpy())
        if (k + 1) % 500 == 0:
            print(f"  {k + 1} / {len(metas)}", flush=True)
    print(f"embedded {len(metas) - len(skipped)} of {len(metas)} folders in {run}")
    assert not skipped, f"{len(skipped)} sanitized texts have no stored LLM2Vec vector, e.g. {skipped[:3]}"


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare")
    s.add_argument("--humanml3d-texts", help="HumanML3D texts/ folder (one .txt per motion)")
    s.add_argument("--extra", nargs="*", help="stream-style prompt files, one prompt per line")
    s.add_argument("--out", required=True, help="for example data/<name>/captions.jsonl")
    s.add_argument("--max-captions", type=int, help="stop after about this many (pilot)")
    s.add_argument("--ardy-prompts", help="fixed prompts for the side-by-side motion check (split 'ardy')")
    s.add_argument("--seed-timelines", help="BONES-SEED timelines.jsonl (nvidia/SEED-Timeline-Annotations)")
    s.add_argument("--seed-splits", help="splits/ folder of nvidia/Kimodo-Motion-Gen-Benchmark")
    s.add_argument("--seed-max-motions", type=int, help="SEED: use only this many motions (smoke test)")
    s.add_argument("--reword", help="JSONL of {source, rewordings} for validation labels (split 'reword')")
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("reword-sources")
    s.add_argument("--captions", required=True)
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--out", required=True, help="JSONL template; fill in 'rewordings', then prepare --reword")
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("seed-joins", help="joined multi-event SEED texts for training (step 3)")
    s.add_argument("--seed-timelines", required=True)
    s.add_argument("--seed-splits", required=True)
    s.add_argument("--out", default="data/join/captions.jsonl")
    s.add_argument("--n-train", type=int, default=20000)
    s.add_argument("--n-val", type=int, default=2000)
    s.add_argument("--exclude", nargs="*", default=["data/full", "data/seed", "data/bench"],
                   help="datasets whose texts are skipped")
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("embed")
    s.add_argument("--captions", required=True, help="for example data/<name>/captions.jsonl")
    s.add_argument("--encoder", choices=("llm2vec", "qwen", "api"), required=True)
    s.add_argument("--provider", help="api: name used in the adapter file name, for example openai")
    s.add_argument("--model", help="api: model ID, for example text-embedding-3-large")
    s.add_argument("--base-url", help="api: OpenAI-compatible base URL, for example https://api.openai.com/v1")
    s.add_argument("--api-key-env", default="EMBEDDING_API_KEY", help="api: env var that holds the key")
    s.add_argument("--settings", default="{}", help='api: extra JSON request fields, for example {"task_type": "..."}')
    s.add_argument("--no-dims-param", action="store_true", help="api: do not send 'dimensions' (fixed-size models)")
    s.add_argument("--repo", help="llm2vec: premerged quantized repo, for example voxta/Llama-3-8B-LLM2Vec-ARDY-NF4")
    s.add_argument("--model-path", help="qwen: local folder or Hugging Face ID, for example Qwen/Qwen3-Embedding-8B")
    s.add_argument("--dims", type=int, help="qwen: MRL output size (default: full)")
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--out", required=True, help="for example data/<name>/emb/qwen3-0.6b_1024")

    data_help = "dataset dirs, each with captions.jsonl and emb/; train splits are pooled"
    select_help = "dataset whose validation makes every choice, by folder name, e.g. seed (default: the last --data)"

    s = sub.add_parser("pilot")
    s.add_argument("--data", nargs="+", required=True, help=data_help)
    s.add_argument("--select", help=select_help)
    s.add_argument("--target", default="llm2vec_bf16")
    s.add_argument("--inputs", nargs="*", default=[], help="provider embedding folders to fit")
    s.add_argument("--compare", nargs="*", default=[], help="LLM2Vec variants used without adapter")
    s.add_argument("--ardy-ckpt", nargs="*", default=[], help="denoiser.safetensors paths or ARDY repo IDs")
    s.add_argument("--lambdas", type=float, nargs="*", default=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    s.add_argument("--out", default="pilot_report.json")
    s.add_argument("--test", action="store_true", help="report on test (use once, at the end)")
    s.add_argument("--curve", type=int, nargs="*", help="ridge learning curve: train subset sizes (all = a large N)")

    s = sub.add_parser("train")
    s.add_argument("--data", nargs="+", required=True, help=data_help)
    s.add_argument("--select", help=select_help)
    s.add_argument("--ardy-ckpt", nargs="*", default=[],
                   help="denoiser.safetensors paths or ARDY repo IDs; keep the checkpoint by val e_c")
    s.add_argument("--target", default="llm2vec_bf16")
    s.add_argument("--input", required=True)
    s.add_argument("--out", required=True, help="eddy_<model>_<dims>.safetensors")
    s.add_argument("--linear-only", action="store_true")
    s.add_argument("--init", choices=("random", "ridge"), default="random")
    s.add_argument("--ridge-lambda", type=float, default=0.01, help="--init ridge: lambda (pilot's choice)")
    s.add_argument("--ardy-loss", type=float, default=0.0, help="weight of the ARDY-space loss term (needs --ardy-ckpt)")
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--wd", type=float, default=1e-4)
    s.add_argument("--batch", type=int, default=512)
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--eval-every", type=int, default=100)
    s.add_argument("--dropout", type=float, default=0.1)
    s.add_argument("--hidden", type=int, default=2048)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int, help="train on the first N rows only (overfit check: 32)")
    s.add_argument("--device", default="cuda")

    s = sub.add_parser("bench-vectors", help="Kimodo benchmark (.venv): stored vectors for every suite text")
    s.add_argument("--suite", required=True, help="built suite root (also used as --prompts by motion-generate)")
    s.add_argument("--adapter", required=True)
    s.add_argument("--name", default="eddy")
    s.add_argument("--data", nargs="+", default=["data/full", "data/seed"])
    s.add_argument("--exclude-to", help="move samples without stored vectors out of the built suite, into here")

    s = sub.add_parser("bench-embed", help="Kimodo benchmark (.venv-kimodo): TMR embeddings from stored vectors")
    s.add_argument("--run", required=True, help="a motion-generate output folder")
    s.add_argument("--suite", required=True, help="the folder with texts.json and vectors/")
    s.add_argument("--device", default="cuda")

    s = sub.add_parser("motion-prompts", help="motion check, step 1 (.venv): prompt folders and vectors")
    s.add_argument("--data", nargs="+", default=["data/full", "data/seed"], help="prompts come from the last")
    s.add_argument("--out", default="runs/motion/prompts")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--input", default="qwen3-8b_4096")
    s.add_argument("--adapter", nargs="*", default=[], help="name=path.safetensors, one vector file each")
    s.add_argument("--ridge-lambda", type=float, default=0.01, help="0: no ridge vectors")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--split", default="val", help="test: use once, at the end of step 4")

    s = sub.add_parser("motion-generate", help="motion check, step 2 (.venv-ardy): ARDY with stored vectors")
    s.add_argument("--prompts", default="runs/motion/prompts")
    s.add_argument("--vectors", required=True, help="a file name in <prompts>/vectors/, without .npy")
    s.add_argument("--seed", type=int, required=True, help="seed set: A = 0, B = 1")
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="soma", help="ARDY model; soma = ARDY-SOMA-RP-30FPS-Horizon60")
    s.add_argument("--ardy-repo", default="ext/ardy")
    s.add_argument("--batch", type=int, default=1, help="keep 1: mixed lengths over ~10 s crash ARDY")

    s = sub.add_parser("motion-score", help="motion check, step 3 (.venv-kimodo): TMR R@3 and FID")
    s.add_argument("--prompts", default="runs/motion/prompts")
    s.add_argument("--runs", nargs="+", required=True, help="generated run folders; the first is the reference")
    s.add_argument("--out")
    s.add_argument("--device", default="cuda")

    a = p.parse_args(argv)
    if a.cmd == "prepare":
        if bool(a.seed_timelines) != bool(a.seed_splits):
            p.error("--seed-timelines and --seed-splits go together")
        prepare(a.humanml3d_texts, a.extra, a.out, a.seed, a.max_captions, a.ardy_prompts,
                a.seed_timelines, a.seed_splits, a.seed_max_motions, a.reword)
    elif a.cmd == "reword-sources":
        reword_sources(a.captions, a.n, a.out, a.seed)
    elif a.cmd == "seed-joins":
        seed_joins(a.seed_timelines, a.seed_splits, a.out, a.n_train, a.n_val, a.exclude, a.seed)
    elif a.cmd == "embed":
        api = None
        if a.encoder == "api":
            if not (a.provider and a.model and a.base_url and a.dims):
                p.error("--encoder api needs --provider, --model, --base-url and --dims")
            api = {"provider": a.provider, "model": a.model, "dims": a.dims, "base_url": a.base_url,
                   "key_env": a.api_key_env, "settings": json.loads(a.settings), "send_dims": not a.no_dims_param}
        embed(a.captions, a.out, a.encoder, a.model_path, a.repo, a.dims, a.batch, api=api)
    elif a.cmd == "pilot":
        pilot(a.data, a.target, a.inputs, a.compare, a.ardy_ckpt, a.lambdas, a.out, a.test, a.select, a.curve)
    elif a.cmd == "train":
        train(a.data, a.target, a.input, a.out, not a.linear_only, a.init, a.lr, a.wd, a.batch,
              a.steps, a.eval_every, a.dropout, a.hidden, a.seed, a.limit, a.device, a.ardy_ckpt, a.select,
              a.ridge_lambda, a.ardy_loss)
    elif a.cmd == "bench-vectors":
        bench_vectors(a.suite, a.data, a.adapter, a.name, exclude_to=a.exclude_to)
    elif a.cmd == "bench-embed":
        bench_embed(a.run, a.suite, a.device)
    elif a.cmd == "motion-prompts":
        motion_prompts(a.data, a.out, a.n, a.input, adapters=a.adapter, ridge_lambda=a.ridge_lambda, seed=a.seed,
                       split=a.split)
    elif a.cmd == "motion-generate":
        motion_generate(a.prompts, a.vectors, a.seed, a.out, a.model, a.ardy_repo, a.batch)
    elif a.cmd == "motion-score":
        motion_score(a.prompts, a.runs, out=a.out, device=a.device)


if __name__ == "__main__":
    sys.exit(main())
