# EDDY: Embedding aDapter for ARDY

**Built with Meta Llama 3.** The published model is named **Llama-3-EDDY-Qwen3-Embedding-8B**, as the [Meta Llama 3 Community License](LICENSE-LLAMA3.txt) requires ([Acceptable Use Policy](USE_POLICY-LLAMA3.md)). The copies are from the `main` branch of github.com/meta-llama/llama3.

EDDY (the project and code name; the model is Llama-3-EDDY-Qwen3-Embedding-8B) maps a Qwen3-Embedding-8B text vector to the LLM2Vec vector that NVIDIA's [ARDY](https://github.com/nv-tlabs/ardy) text-to-motion model expects. With EDDY, ARDY runs without its 8B LLM2Vec text encoder (about 14 GB of VRAM in bf16, and no provider hosts it). You embed the prompt with any host of Qwen3-Embedding-8B (local, OpenRouter, or another API), and EDDY turns that vector into the one ARDY needs. The adapter is one linear map (ridge regression, about 67 MB), fit on about 112k caption pairs.

Scope: `ARDY-SOMA-RP-30FPS-Horizon60` and Qwen3-Embedding-8B (4096 dimensions, no instruction). Other ARDY checkpoints and other embedding models are not supported by the released adapter.

> EDDY is an independent project. It is not made by or endorsed by NVIDIA.

## How it works

```
text -> embeddings API (Qwen3-Embedding-8B, 4096-d) -> EDDY (linear map) -> vector [1, 4096] -> ARDY
```

ARDY uses one pooled 4096-d vector per prompt, so a translation between two embedding spaces can replace the encoder. LLM2Vec stays the teacher: EDDY is trained so that its output is close to the bf16 LLM2Vec vector of the same text. The goal is the same motion, not only the same vector, so the adapter was chosen by running ARDY and scoring the motions, not by vector error. An empty prompt becomes an exact zero vector and is never sent to the provider, because ARDY's classifier-free guidance needs a zero.

## Quick start

```bash
git clone https://github.com/Xyri1/EDDY && cd EDDY
pip install -r requirements.txt
hf download GhoXtie/Llama-3-EDDY-Qwen3-Embedding-8B eddy_qwen3-embedding-8b_4096.safetensors --local-dir .
```

You also need ARDY itself ([install steps](https://github.com/nv-tlabs/ardy)) and its checkpoint, which is downloaded from Hugging Face on first use. Then:

```python
import os
from eddy_encoder import APIEmbedder, EddyTextEncoder
from ardy.model.load_model import load_model

os.environ["EMBEDDING_API_KEY"] = "<your key>"   # or set it in your shell
emb = APIEmbedder("openrouter", "qwen/qwen3-embedding-8b", 4096, "https://openrouter.ai/api/v1")
enc = EddyTextEncoder(emb, "eddy_qwen3-embedding-8b_4096.safetensors", device="cuda")
model = load_model("ARDY-SOMA-RP-30FPS-Horizon60", text_encoder=enc)
# Use `model` as in ARDY's README: the text encoder is already attached.
```

`APIEmbedder` works with any OpenAI-compatible `/embeddings` endpoint. Use the host's own model name. For a local model, use `QwenEmbedder("Qwen/Qwen3-Embedding-8B", 4096)` from the same file (weights load from Hugging Face), or any callable that maps a list of texts to an array of shape `[B, 4096]`.

**You must serve the right model.** EDDY does not check which model is behind your endpoint. It only refuses a vector of the wrong size. Serve `Qwen/Qwen3-Embedding-8B` with 4096 dimensions and no instruction prefix. Another model with 4096 dimensions would run without an error and give bad motion.

## Demo

A local Gradio demo with ARDY's viewer is in [`demo/`](demo/README.md). You set an OpenAI-compatible embedding endpoint, an API key and a model name; the adapter is pulled from Hugging Face; ARDY runs on a GPU.

## Model

| Model | Adapter file | Input model | Dims | Target | ARDY checkpoint | Size | Hugging Face |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3-EDDY-Qwen3-Embedding-8B | `eddy_qwen3-embedding-8b_4096.safetensors` | Qwen3-Embedding-8B (any host) | 4096 | LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised (bf16) | ARDY-SOMA-RP-30FPS-Horizon60 | 67 MB | [GhoXtie/Llama-3-EDDY-Qwen3-Embedding-8B](https://huggingface.co/GhoXtie/Llama-3-EDDY-Qwen3-Embedding-8B) |

The file header records the input model, the dimensions and the LLM2Vec checkpoint of the training vectors.

## Results

Motion is scored with ARDY-SOMA-RP-30FPS-Horizon60 and the TMR-SOMA-RP-v1 evaluator. All numbers are R@3 (higher is better) against the same model driven by bf16 LLM2Vec, with the same seeds. Full protocol and tables: [docs/report.md](docs/report.md).

| Check | LLM2Vec | EDDY | Difference |
| --- | --- | --- | --- |
| 500 SEED validation prompts, 5 seeds, local Qwen | 79.1 ± 2.1 | 78.1 ± 1.0 | −1.0 ± 2.2 (paired) |
| Same prompts, Qwen from OpenRouter | 79.1 ± 2.1 | 77.2 ± 1.6 | −1.9 ± 1.1 (paired) |
| 500 joined multi-event validation prompts | 87.2 ± 1.4 | 84.9 ± 0.8 | −2.3 |
| Kimodo benchmark, timeline single (earlier ridge adapter) | 64.78 | 64.34 | −0.4 |
| Kimodo benchmark, overview (earlier ridge adapter) | 75.90 | 73.94 | −2.0 (real) |
| Kimodo benchmark, timeline multi (final adapter) | 77.19 | 73.38 | −3.8 (real) |

On the SEED validation prompts the loss is within the seed noise. On the benchmark it is about 0 to 4 R@3 points, largest on multi-event prompts. The benchmark was run once for the earlier adapter and once more for timeline multi after the final adapter was chosen (a second look, not used for the choice). Running ARDY end to end with `EddyTextEncoder` on 45 prompts gave R@3 94.2 ± 3.0 (EDDY) and 95.1 ± 1.7 (LLM2Vec). In a visual check of those 45 motions, no pair had a clearly better side.

## Known limits

- **Multi-event prompts.** Prompts that join several events lose the most (benchmark timeline multi, −3.8 R@3 after training on joined texts, −6.0 before).
- **Prompt style.** The adapter is trained and checked on texts in the style of ARDY's training labels: third person, one or two full sentences ("A person walks forward and waves."). Imperatives, "I" or "you", single words, negation and durations in the text were not in those labels and are out of scope. How much this changes the motion was not measured.
- **Fine hand detail.** The worst vector errors on stream-style prompts are finger-level signs ("makes an OK sign").
- **Hosted Qwen is at the edge.** Hosted (OpenRouter) vectors differ slightly from local ones (cosine 0.996 to 0.999). The motion check passes, but R@3 against LLM2Vec was lower in all 5 seeds.
- **Latency.** One prompt through OpenRouter took a median of 3.7 s (maximum 38 s over 10 calls) from the author's machine. Local Qwen takes 0.09 s but needs about 14 GB of VRAM, so it does not fit beside ARDY on a 16 GB GPU.
- **CPU.** Single runs after warm-up on a Ryzen 9 7900X (from the demo): one 6 s motion took 2.8 s on CPU and 0.4 to 0.7 s on an RTX 4080 SUPER. The first ARDY load took 1.2 s on CPU and 10 to 14 s on GPU (weights cached). An embedding request through OpenRouter took 1.4 to 2.2 s. These are single runs, not a benchmark.
- **One checkpoint, one input model.** A new input model needs a new adapter (training takes minutes once the embeddings exist).

## Training and evaluation

`eddy.py` is the training and evaluation CLI (`prepare`, `reword-sources`, `seed-joins`, `embed`, `pilot`, `train`, `motion-prompts`, `motion-generate`, `motion-score`). `test_eddy.py` is a synthetic end-to-end self-check: run `python test_eddy.py` after every change.

The training data is not included. Rebuild it from:

- HumanML3D captions ([HumanML3D](https://github.com/EricGuo5513/HumanML3D));
- [SEED-Timeline-Annotations](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations) and the `splits/` files of the [Kimodo Motion Generation Benchmark](https://huggingface.co/datasets/nvidia/Kimodo-Motion-Gen-Benchmark) (neither needs the gated BONES-SEED motions; the benchmark motion cases do);
- LLM2Vec targets: made with ARDY's own bf16 LLM2Vec, alone on the GPU, one prompt per call;
- Qwen3-Embedding-8B vectors of the same texts.

The adapter used for the results is: `python eddy.py train --data data/full data/seed data/join --select seed --linear-only --init ridge --steps 0` (ridge, λ = 0.01). Details: [docs/report.md](docs/report.md#3-data).

## Citation

See [CITATION.cff](CITATION.cff). Please also cite ARDY, which this work depends on:

```bibtex
@article{zhao2026ardy,
  title   = {ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation},
  author  = {Zhao, Kaifeng and Petrovich, Mathis and Zhang, Haotian and Wang, Tingwu and Tang, Siyu and Rempe, Davis},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2026},
  volume  = {45},
  number  = {4},
  articleno = {86},
  doi     = {10.1145/3811284}
}
```

## License

**Apache License 2.0** ([LICENSE](LICENSE)) for EDDY's code, the adapter weights and the documentation. Attributions are in [NOTICE](NOTICE); keep it when you redistribute EDDY, as the Apache License requires. The adapter contains no Llama 3 weights, and on the author's reading the Meta Llama 3 Community License asks only for the "Llama 3" name prefix and the attribution above. If its terms are found to apply to the adapter, they apply in addition to the Apache License.

## Third-party licenses and attribution

This is the author's reading of the licenses, not legal advice. Checked at the source on 2026-10-29.

**Built with Meta Llama 3.** EDDY's training targets were produced by LLM2Vec, which runs Meta Llama 3 weights. The model name therefore starts with "Llama 3" (Meta Llama 3 Community License, section 1.b.i). The agreement and the use policy are in this repo: [LICENSE-LLAMA3.txt](LICENSE-LLAMA3.txt), [USE_POLICY-LLAMA3.md](USE_POLICY-LLAMA3.md). The copies are from the `main` branch of github.com/meta-llama/llama3.

Training data includes Motion Data by Bones Studio. EDDY was not trained on Bones motion data. The motions were used only to run the benchmark evaluation. The training texts come from NVIDIA's SEED-Timeline-Annotations (CC BY 4.0).

| Work | Use in EDDY | License |
| --- | --- | --- |
| Meta Llama 3 (inside LLM2Vec) | Runs inside the teacher that made the training targets | Meta Llama 3 Community License (copy in [LICENSE-LLAMA3.txt](LICENSE-LLAMA3.txt)) |
| [LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised) and [-mntp](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp) | Teacher: EDDY was trained on its output vectors | MIT (Hugging Face model card). It covers McGill's adapter weights. The Llama 3 base stays under Meta's license. |
| [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Input model; training vectors. Not redistributed. | Apache-2.0 (Hugging Face model card) |
| [ARDY-SOMA-RP-30FPS-Horizon60](https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-Horizon60) | Run-time model and evaluation. The demo downloads it at run time. Not redistributed. | NVIDIA Open Model Agreement. NVIDIA claims no ownership of outputs. |
| [ARDY](https://github.com/nv-tlabs/ardy) code | Imported by the demo; not vendored | Apache-2.0 |
| [TMR-SOMA-RP-v1](https://huggingface.co/nvidia/TMR-SOMA-RP-v1) | Motion scoring in evaluation only. Not redistributed. | NVIDIA Open Model License. Its use is subject to NVIDIA's Trustworthy AI terms. |
| [SEED-Timeline-Annotations](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations) | Training and validation texts. EDDY's joined texts are modified (concatenated) versions of these labels. | CC BY 4.0 |
| [Kimodo Motion Generation Benchmark](https://huggingface.co/datasets/nvidia/Kimodo-Motion-Gen-Benchmark) | Benchmark protocol and results | CC BY 4.0 |
| [HumanML3D](https://github.com/EricGuo5513/HumanML3D) | Only its captions were used; no AMASS motion data | MIT |
| [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) motion data | Only to run the Kimodo benchmark evaluation; never for training; not redistributed | [Bones SEED license](https://bones.studio/info/seed-license) (gated). The project is non-profit, under the license's non-commercial research terms. The license counts statistical findings and publications as "Results". |

