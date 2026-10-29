# EDDY: Embedding aDapter for ARDY

Run NVIDIA's [ARDY](https://github.com/nv-tlabs/ardy) text-to-motion model without loading its 8B LLM2Vec text encoder. EDDY maps Qwen3-Embedding-8B embeddings into the vector space ARDY expects, so you can get text embeddings from an API and generate motion locally.

The adapter is a single linear map, trained with ridge regression on about 112,000 caption pairs. It is about 67 MB. With the embedding model off the GPU, ARDY and EDDY peaked at **0.96 GiB of PyTorch-allocated VRAM** in our batch-size-one evaluation. CUDA overhead and reserved memory are additional. See the [technical report](docs/report.md#end-to-end-and-hosted-qwen) for the measured setup.

**Built with Meta Llama 3.** The released model is [Llama-3-EDDY-Qwen3-Embedding-8B](https://huggingface.co/GhoXtie/Llama-3-EDDY-Qwen3-Embedding-8B). See [licensing and attribution](#license) below.

EDDY is an independent project. It is not made by or endorsed by NVIDIA.

## How it works

```text
Prompt → Qwen3-Embedding-8B → EDDY → ARDY → Motion
         local or API       local   local
```

ARDY takes one 4096-dimensional text vector per prompt. EDDY learns to map a Qwen embedding to the bf16 LLM2Vec vector for the same text. We evaluate the resulting motions to decide whether the adapter works; vector error is a secondary check.

Empty prompts return an exact zero vector for ARDY's classifier-free guidance. They are never sent to the embedding provider.

The released adapter supports **ARDY-SOMA-RP-30FPS-Horizon60** with **Qwen3-Embedding-8B**, using 4096 dimensions and no instruction prefix.

## Quick start

Install [ARDY](https://github.com/nv-tlabs/ardy) following its setup instructions, then clone EDDY and download the adapter:

```bash
git clone https://github.com/Xyri1/EDDY
cd EDDY
pip install -r requirements.txt
hf download GhoXtie/Llama-3-EDDY-Qwen3-Embedding-8B eddy_qwen3-embedding-8b_4096.safetensors --local-dir .
```

Copy `.env.example` to `.env` in the EDDY repository and set your embedding provider's base URL, model ID, and API key. For OpenRouter:

```dotenv
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
EMBEDDING_API_KEY=your-api-key
```

`.env` is ignored by Git. Run the following example from the repository root:

```python
import os

from dotenv import load_dotenv
from eddy_encoder import APIEmbedder, EddyTextEncoder
from ardy.model.load_model import load_model

load_dotenv(".env", encoding="utf8")

emb = APIEmbedder(
    "api",
    os.environ["EMBEDDING_MODEL"],
    4096,
    os.environ["EMBEDDING_BASE_URL"],
    key_env="EMBEDDING_API_KEY",
)
enc = EddyTextEncoder(
    emb,
    "eddy_qwen3-embedding-8b_4096.safetensors",
    device="cuda",
)
model = load_model("ARDY-SOMA-RP-30FPS-Horizon60", text_encoder=enc)
# Continue with ARDY's generation workflow using this model.
```

ARDY downloads its checkpoint from Hugging Face on first use.

`APIEmbedder` accepts any OpenAI-compatible embeddings endpoint. Use the provider's model name for Qwen3-Embedding-8B. EDDY checks the vector size, but cannot verify which model produced it. A different model with the same dimensions will pass that check and produce incorrect conditioning.

For local embeddings, use `QwenEmbedder("Qwen/Qwen3-Embedding-8B", 4096)` from `eddy_encoder.py`. You can also supply a callable that takes a list of texts and returns Qwen3-Embedding-8B vectors with shape `[B, 4096]`.

## Demo

The [Gradio demo](demo/README.md) lets you enter a prompt and view the generated motion in ARDY's 3D viewer. It supports GPU and CPU generation.

After installing the [demo dependencies](demo/README.md#install), run:

```bash
python demo/app.py
```

Open <http://127.0.0.1:7860> and generate a motion. The demo reads the base URL, model ID, and API key from the repository's `.env` file and downloads the adapter automatically. You can override these settings in the page. Existing environment variables take precedence over `.env`.

## Adapter

Download [Llama-3-EDDY-Qwen3-Embedding-8B from Hugging Face](https://huggingface.co/GhoXtie/Llama-3-EDDY-Qwen3-Embedding-8B).

| Property | Value |
| --- | --- |
| File | `eddy_qwen3-embedding-8b_4096.safetensors` |
| Size | About 67 MB |
| Architecture | Linear map; ridge regression, λ = 0.01 |
| Input | Qwen3-Embedding-8B, 4096 dimensions, no instruction prefix |
| Target | LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised, bf16 |
| Supported ARDY checkpoint | ARDY-SOMA-RP-30FPS-Horizon60 |

The adapter works with local or hosted Qwen3-Embedding-8B vectors. Its safetensors header records the input model, dimensions, provider, dtype, and target checkpoint used for training.

## Results

We compare ARDY driven by EDDY against ARDY driven by bf16 LLM2Vec, using the same seeds. Scores below are R@3 from the TMR-SOMA-RP-v1 evaluator; higher is better. The [technical report](docs/report.md) gives the full protocol and results.

| Evaluation | LLM2Vec | EDDY | Difference |
| --- | --- | --- | --- |
| 500 SEED validation prompts, 5 seeds, local Qwen | 79.1 ± 2.1 | 78.1 ± 1.0 | −1.0 ± 2.2 (paired) |
| Same prompts, Qwen from OpenRouter | 79.1 ± 2.1 | 77.2 ± 1.6 | −1.9 ± 1.1 (paired) |
| 500 joined multi-event validation prompts | 87.2 ± 1.4 | 84.9 ± 0.8 | −2.3 |
| Kimodo benchmark, timeline single (earlier adapter) | 64.78 | 64.34 | −0.4 |
| Kimodo benchmark, overview (earlier adapter) | 75.90 | 73.94 | −2.0 |
| Kimodo benchmark, timeline multi (released adapter) | 77.19 | 73.38 | −3.8 |

On SEED validation, the difference is within the observed seed variation. Benchmark scores are about 0 to 4 R@3 points lower, with the largest gap on multi-event prompts. The released adapter was not evaluated on the benchmark's overview or timeline-single cases.

The benchmark was run once with the earlier adapter. Timeline multi was evaluated again after the released adapter was selected; that second result was not used to make the selection.

An end-to-end run on 45 prompts scored 94.2 ± 3.0 with EDDY and 95.1 ± 1.7 with LLM2Vec. A visual comparison found no clear preference between the two sets of motions.

## Limitations and performance

- **Multi-event prompts:** the largest measured gap is on benchmark timeline multi, at −3.8 R@3 for the released adapter.
- **Prompt style:** evaluation covers third-person descriptions such as “A person walks forward and waves.” The effect of imperatives, first- or second-person prompts, single words, negation, and durations in the text has not been measured.
- **Hand detail:** finger-level signs, such as “makes an OK sign,” have some of the largest vector errors on stream-style prompts.
- **Hosted embeddings:** OpenRouter vectors differ slightly from local Qwen vectors (cosine similarity 0.996–0.999). They passed the motion check, although R@3 was lower than LLM2Vec in all five seeds. Other hosts have not been tested.
- **API latency:** a 10-call OpenRouter check had a median of 3.7 s and a maximum of 38 s from the author's machine. Separate demo measurements took 1.4–2.2 s per embedding request. Local Qwen took 0.09 s per prompt but used about 14 GB of VRAM.
- **Generation time:** after warm-up, a 6 s motion took 0.4–0.7 s on an RTX 4080 SUPER and 2.8 s on a Ryzen 9 7900X CPU. Initial ARDY loading took 10–14 s on GPU and 1.2 s on CPU, with weights cached. These are individual demo measurements, not a benchmark.
- **Compatibility:** only the ARDY checkpoint and embedding model listed above are supported. A different input model needs a separately trained adapter.

## Training and evaluation

`eddy.py` provides the data preparation, embedding, training, and motion evaluation commands. Run `python eddy.py --help` for the command list. `test_eddy.py` is a synthetic end-to-end self-check; run `python test_eddy.py` after changes to `eddy.py` or `eddy_encoder.py`.

Training data and embeddings are not included. The adapter was trained on HumanML3D captions, SEED labels, and joined multi-event texts, paired with Qwen3-Embedding-8B and bf16 LLM2Vec vectors. The [report](docs/report.md#3-data) describes the sources and grouped splits. Rebuilding the text pairs does not require gated BONES-SEED motions; running the motion benchmark does.

With the prepared datasets, the released adapter is fit using:

```bash
python eddy.py train --data data/full data/seed data/join --select seed --linear-only --init ridge --steps 0
```

This fits ridge regression with λ = 0.01. SEED validation determines the selection.

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

