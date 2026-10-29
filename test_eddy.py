"""End-to-end self-check of eddy.py on synthetic data (no GPU, no ARDY). Run: python test_eddy.py"""

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

import eddy


def write_emb(folder, x, captions):
    folder.mkdir(parents=True)
    for s in range(0, len(x), eddy.SHARD):
        np.save(folder / f"part_{s // eddy.SHARD:05d}.npy", x[s: s + eddy.SHARD].astype(np.float32))
    meta = {"n": len(x), "captions_sha256": eddy.file_sha256(captions), "provider": "local",
            "model": "fake", "dims": str(x.shape[1]), "dtype": "float32", "instruction": "", "settings": "{}"}
    (folder / "meta.json").write_text(json.dumps(meta))


def fake_seed(tmp):
    """Synthetic BONES-SEED timelines + benchmark splits. Groups Walk_K; K % 10 == 0: content test;
    K % 10 == 1: motion _002 is in the repetition test; group 22 has one train and one content-test motion."""
    motions, split = [], {}
    for k in range(60):
        for take, actor in ((1, 1), (2, 2)):
            for mirror in ("", "_M"):
                name = f"Walk {k}_{take:03d}__A{actor:02d}{mirror}" if k == 5 else f"Walk_{k}_{take:03d}__A{actor:02d}{mirror}"
                side = ("left", "right")[(take + (mirror == "_M")) % 2]
                motions.append({"filename": name, "overview_description": f"a person walks pattern {k}",
                                "events": [{"start_time": 0.0, "end_time": 1.5, "description": f"a person {k} turns {side}"},
                                           {"start_time": 1.5, "end_time": 4.0, "description": f"a person {k} take {take} waves"}],
                                "propagated_from_filename": None})
                split[name] = ("test_content" if k % 10 == 0 or (k == 22 and take == 2) else
                               "test_repetition" if k % 10 == 1 and take == 2 else "train")
    # Shared texts: the same label on a train motion and on a held-out motion.
    motions[0]["events"].append({"start_time": 4, "end_time": 5, "description": "a person stands idle"})  # Walk_0: test
    motions[8]["events"].append({"start_time": 4, "end_time": 5, "description": "a person stands idle"})  # Walk_2: train
    rep = next(m for m in motions if split[m["filename"]] == "test_repetition")
    rep["events"].append({"start_time": 4, "end_time": 5, "description": "a person bows"})
    motions[-1]["events"].append({"start_time": 4, "end_time": 5, "description": "a person bows"})  # train
    motions[-1]["events"].append({"start_time": 5, "end_time": 6,  # leaked annotation template: dropped
                                  "description": "Bows overview description a person bows. Event 1 detail description"})
    (tmp / "timelines.jsonl").write_text("".join(json.dumps(m) + "\n" for m in motions))
    (tmp / "splits").mkdir()
    for s in ("train", "test_content", "test_repetition"):
        (tmp / "splits" / f"{s}_split_paths.txt").write_text(
            "".join(f"220705/{n}\n" for n, v in split.items() if v == s))
    return motions, split


def check_seed(tmp):
    """prepare with SEED labels: group split, benchmark texts held out, durations, rewordings."""
    motions, split = fake_seed(tmp)
    assert eddy.seed_content("Idle_Left_001__A017_M") == "Idle_Left"
    cap = tmp / "seed" / "captions.jsonl"
    args = dict(texts_dir=None, extra=None, out=cap, seed=0, seed_timelines=tmp / "timelines.jsonl",
                seed_splits=tmp / "splits")
    eddy.prepare(**args)
    rows = eddy.load_captions(cap)
    by_text = {r["text"]: r for r in rows}
    assert not any("description" in t for t in by_text), "leaked template labels must be dropped"
    assert by_text["a person stands idle"]["split"] == "test", "a content-test text must never be trained on"
    assert by_text["a person bows"]["split"] == "test_rep", "a repetition-test text must never be trained on"
    assert all(by_text[f"a person {k} turns left"]["split"] == "test" for k in (0, 22)), "whole group goes to test"
    held_motion_texts = {e["description"] for m in motions if split[m["filename"]] != "train" for e in m["events"]}
    assert not any(r["split"] in ("train", "val") for t in held_motion_texts for r in [by_text[t]])
    by_group = {}
    for r in rows:
        if r["split"] in ("train", "val", "test"):
            by_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values()), "a SEED content group crosses train / val / test"
    assert {"train", "val", "test", "test_rep"} <= {r["split"] for r in rows}
    assert by_text["a person 2 turns left"]["duration"] == 1.5 and "duration" not in by_text["a person walks pattern 2"]

    # Rewordings: template from val, filled in, appended without changing earlier row ids.
    tpl = tmp / "reword.jsonl"
    eddy.reword_sources(cap, 3, tpl)
    items = [json.loads(l) for l in tpl.read_text().splitlines()]
    assert len(items) == 3 and all(by_text[i["source"]]["split"] == "val" for i in items)
    for i in items:
        i["rewordings"] = [f"{i['source']} (rewording {j})" for j in range(2)] + [i["source"]]  # last one: duplicate
    tpl.write_text("".join(json.dumps(i) + "\n" for i in items))
    eddy.prepare(**args, reword=tpl)
    rows2 = eddy.load_captions(cap)
    assert rows2[: len(rows)] == rows, "rewordings must not change existing rows"
    rw = [r for r in rows2 if r["split"] == "reword"]
    assert len(rw) == 6 and all(rows2[r["source"]]["split"] == "val" for r in rw)

    # Joined multi-event texts: same split as their motion's labels, no held-out motion, order kept.
    jcap = tmp / "join" / "captions.jsonl"
    eddy.seed_joins(tmp / "timelines.jsonl", tmp / "splits", jcap, n_train=1000, n_val=1000, exclude=[tmp / "seed"])
    joins = eddy.load_captions(jcap)
    group_split = {r["group"]: r["split"] for r in rows if r["split"] in ("train", "val")}
    assert joins and {r["split"] for r in joins} == {"train", "val"}
    assert all(group_split[r["group"]] == r["split"] for r in joins), "a joined text left its group's split"
    both = [r for r in joins if " turns " in r["text"] and " waves" in r["text"]]
    assert both and all(r["text"].index(" turns ") < r["text"].index(" waves") and r["duration"] >= 4.0 for r in both)
    assert not any(f"person {k} take 2" in r["text"] for k in range(1, 60, 10) for r in joins), "a test_rep motion"
    rng = random.Random(0)
    j = [eddy._join(["A person walks forward.", "A person waves with the right hand."], rng) for _ in range(40)]
    assert {"A person walks forward, then waves with the right hand.",
            "A person walks forward. Then, a person waves with the right hand.",
            "A person walks forward. Next, the person waves with the right hand."} <= set(j), set(j)
    assert eddy._join(["A person walks.", "A person is standing."], rng).count("person") == 2, "no 'then is standing'"
    return rows2


def main():
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())

    # --- prepare: fake HumanML3D folder with mirrored files
    texts = tmp / "texts"
    texts.mkdir()
    for i in range(400):
        side = "left" if i % 4 == 0 else "right"
        (texts / f"{i:06d}.txt").write_text(f"a person {i} turns {side}#tok#0#0\nperson {i} walks#tok#0#0\n")
        (texts / f"M{i:06d}.txt").write_text(f"a person {i} turns {'right' if side == 'left' else 'left'}#t#0#0\n")
    (tmp / "stream.txt").write_text("\n".join(f"stream prompt {i}" for i in range(20)))
    hml = tmp / "hml"
    cap = hml / "captions.jsonl"
    (tmp / "ardy.txt").write_text("a person 0 turns left\nardy only prompt\n")  # first line duplicates a caption
    eddy.prepare(texts, [tmp / "stream.txt"], cap, seed=0, ardy_prompts=tmp / "ardy.txt")
    rows = eddy.load_captions(cap)
    ardy = [r["text"] for r in rows if r["split"] == "ardy"]
    assert ardy == ["a person 0 turns left", "ardy only prompt"], ardy
    assert not any(r["text"] == "a person 0 turns left" and r["split"] == "train" for r in rows)
    by_group = {}
    for r in rows:
        if r["split"] in ("train", "val", "test"):
            by_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values()), "a motion group crosses splits"
    seed_rows = check_seed(tmp)
    seed = tmp / "seed"

    d = eddy.Data([hml, seed])
    assert len(d.rows) == len(rows) + len(seed_rows) and d.rows[len(rows)]["id"] == len(rows)
    for src in ("hml", "seed"):
        v = d.view(src)
        assert len(v.pairs) > 0 and not set(v.pairs.ravel()) & set(d.train), "pairs must be held out"
        assert {d.rows[i]["src"] for i in v.pairs.ravel()} == {src}
    assert len(d.reword) == 6 and all(d.rows[s]["split"] == "val" and d.rows[s]["src"] == "seed"
                                      for s in d.reword[:, 1]), "reword sources must be offset into seed rows"

    # --- synthetic spaces: shared latent h; target = big mean offset + linear map; input = other map + noise
    n, k = len(d.rows), 48
    h = rng.normal(size=(n, k))
    h[d.reword[:, 0]] = h[d.reword[:, 1]] + 0.05 * rng.normal(size=(len(d.reword), k))  # rewording ≈ source
    y = 5.0 + h @ rng.normal(size=(k, 4096)) * 0.1
    x = h @ rng.normal(size=(k, 256)) + 0.05 * rng.normal(size=(n, 256))
    for name, v in (("llm2vec_bf16", y), ("fake", x), ("llm2vec_noisy", y + 0.01 * rng.normal(size=y.shape))):
        write_emb(hml / "emb" / name, v[: len(rows)], cap)
        write_emb(seed / "emb" / name, v[len(rows):], seed / "captions.jsonl")
    ckpt = tmp / "denoiser.safetensors"
    save_file({"root.embed_text.weight": torch.randn(512, 4096), "root.embed_text.bias": torch.zeros(512),
               "body.embed_text.weight": torch.randn(512, 4096)}, str(ckpt))

    # --- pilot: per-dataset rows, choices on the last dataset, ε from rewordings
    eddy.pilot([hml, seed], "llm2vec_bf16", ["fake"], ["llm2vec_noisy"], [str(ckpt)],
               [1e-3, 1e-1, 10], tmp / "pilot.json")
    report = json.loads((tmp / "pilot.json").read_text())
    rep = report["rows"]
    assert report["select"] == "seed"
    for src in ("hml", "seed"):
        floor = rep[f"mean (floor) [{src}]"]
        assert floor["R@1"] <= 2 / floor["gallery"], "mean baseline must be at chance"
        assert rep[f"fake ridge [{src}]"]["R@1"] > 0.9, rep[f"fake ridge [{src}]"]["R@1"]
        assert rep[f"fake procrustes [{src}]"]["R@1"] > floor["R@1"]  # 256 -> 4096 works
        assert rep[f"llm2vec_noisy (no adapter) [{src}]"]["R@1"] > 0.99
    assert "stream_R@1" in rep["fake ridge [hml]"] and "stream_R@1" not in rep["fake ridge [seed]"]
    e = next(iter(rep["fake ridge [seed]"]["ardy"].values()))
    assert e["e_median"] < e["e_c_median"], "e (with mean offset) must look smaller than e_c"
    eps = report["epsilon"]
    assert eps["n_rewordings"] == 6 and 0 < eps["eps"] <= eps["eps95"] < 0.5, eps

    # --- train: linear-only from random init must approach ridge (kept by validation e_c on seed)
    out = tmp / "eddy_local_fake_256.safetensors"
    eddy.train([hml, seed], "llm2vec_bf16", "fake", out, residual=False, lr=3e-3,
               steps=1500, eval_every=250, device="cpu", ardy_ckpt=[str(ckpt)])
    model, meta = eddy.load_adapter(out)
    ridge_ec = eddy.worst(rep["fake ridge [seed]"]["ardy"])
    assert meta["val_metric"] == "e_c_median" and meta["select"] == "seed"
    assert float(meta["val_score"]) < 1.5 * ridge_ec, (meta["val_score"], ridge_ec)
    assert set(json.loads(meta["captions_sha256"])) == {"hml", "seed"}

    # --- residual model: MLP starts at zero, so step 0 equals the linear branch
    m = eddy.build_model(256, y.mean(0), y.std(0))
    xt = torch.randn(4, 256)
    assert torch.equal(m.standardized(xt), m.linear(torch.nn.functional.normalize(xt, dim=-1)))

    # --- drop-in encoder: any embedder callable (no metadata), from any host
    def fake_embedder(texts):
        return np.stack([x[[r["text"] for r in rows].index(t)] for t in texts])

    enc = eddy.EddyTextEncoder(fake_embedder, out, device="cpu").eval()
    t = rows[d.view("hml", "val").held[0]]["text"]  # hml rows come first, so ids index `rows` and `x`
    feats, lengths = enc([t, "", "  ", t])
    assert feats.shape == (4, 1, 4096) and lengths == [1, 1, 1, 1]
    assert torch.count_nonzero(feats[1]) == 0 and torch.count_nonzero(feats[2]) == 0, "empty prompt must be zero"
    assert torch.equal(feats[0], feats[3]) and len(enc.cache) == 1
    assert enc.meta["model"] == "fake" and enc.meta["provider"] == "local"  # recorded, not checked
    try:
        eddy.EddyTextEncoder(lambda texts: np.zeros((len(texts), 1024)), out, device="cpu")([t])
        raise AssertionError("a wrong vector size must fail")
    except ValueError as err:
        assert "1024-d" in str(err)
    check_api()
    print("\nALL CHECKS PASSED")


def check_api():
    """APIEmbedder against a local fake endpoint: request body, key header, order by index, retry on 429."""
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.headers["Authorization"], body))
            if len(seen) == 1:  # first call: rate limited
                self.send_response(429)
                self.end_headers()
                return
            data = [{"index": i, "embedding": [float(i)] * body["dimensions"]} for i in range(len(body["input"]))]
            out = json.dumps({"data": data[::-1]}).encode()  # reversed: the client must sort by index
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["FAKE_KEY"] = "test-key"
    emb = eddy.APIEmbedder("fake", "m-1", 8, f"http://127.0.0.1:{srv.server_port}/v1", key_env="FAKE_KEY",
                           settings={"task_type": "SEMANTIC_SIMILARITY"})
    v = emb(["a", "b", "c"])
    srv.shutdown()
    assert v.shape == (3, 8) and v[2, 0] == 2.0, "rows must follow the input order"
    auth, body = seen[-1]
    assert len(seen) == 2 and auth == "Bearer test-key"
    assert body == {"model": "m-1", "input": ["a", "b", "c"], "encoding_format": "float",
                    "task_type": "SEMANTIC_SIMILARITY", "dimensions": 8}
    assert json.loads(emb.meta["settings"]) == {"task_type": "SEMANTIC_SIMILARITY"}

    # --- motion check: stored-vector text encoder (ARDY interface), empty = exact zero, unknown = error
    enc = eddy.StoredTextEncoder(["a b.", "c d."], np.ones((2, 4096), np.float32) * [[1], [2]], device="cpu")
    feat, lengths = enc(["c d.", "", "a b."])
    assert tuple(feat.shape) == (3, 1, 4096) and lengths == [1, 1, 1]
    assert feat[0, 0, 0] == 2 and not feat[1].any() and feat[2, 0, 0] == 1
    try:
        enc(["never stored."])
        raise AssertionError("unknown text must fail")
    except KeyError:
        pass
    assert eddy._sanitize_invariant("A person walks forward.")
    assert not eddy._sanitize_invariant("A person walks. Then sits.") and not eddy._sanitize_invariant("a T-pose.")


if __name__ == "__main__":
    main()
