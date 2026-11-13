# ContextSLM4AVR

The study compares strategies for LLM-based Java vulnerability repair on a fixed hardware budget (a single 80 GB GPU) using OpenAI's open-weight models: running a larger model (`gpt-oss-120b`), LoRA fine-tuning a smaller model (`gpt-oss-20b`), Retrieval-Augmented Generation (RAG), and their combinations. All configurations are evaluated on 42 single-file vulnerabilities from [VUL4J](https://github.com/tuhh-softsec/vul4j), with 10 independent runs each.

The package contains the fine-tuning and inference scripts, the fine-tuning datasets, the prompts and retrieved RAG examples, the raw model outputs of every run, and the per-run test results.

## Contents

```
.
├── finetune_gpt.py        # LoRA fine-tuning of gpt-oss-20b
├── gpt_test_rag.py        # Inference with the 20B models (base / fine-tuned, with or without RAG)
├── gpt_test_120b.py       # Inference with gpt-oss-120b via vLLM (with or without RAG)
├── datasets/              # Fine-tuning data (vulnerable-fixed Java code pairs)
├── prompts/               # VUL4J repair prompts and retrieved RAG examples
├── runs/                  # Raw model outputs, 10 runs per configuration
└── results/               # Test results per vulnerability and run
```

### `datasets/`

| File | Pairs | Source |
|---|---|---|
| `megavul_vul_fixed_pairs_pairs.jsonl` | 2,417 | MegaVul (Java), real-world vulnerability-fixing commits |
| `cybernative.jsonl` | 418 | CyberNative Code Vulnerability and Security dataset (synthetic) |

Both files come from the RAG database of Antal et al. (2026), in which entries overlapping with or highly similar to VUL4J were removed. Each line is a JSON object with the fields `id`, `code_vul` (vulnerable code), and `code_sec` (fixed code); MegaVul entries additionally contain `cve_id` and `cwe_ids`, CyberNative entries `vul_type`.

### `prompts/`

- **`prompts.json`**: the 42 VUL4J repair prompts (`id`, `prompt`). The prompts are the best-performing variant from Antal et al. (2025): CVE-based repair guidelines combined with project-specific context.
- **`rag.json`**: the retrieved examples for each VUL4J vulnerability (`id`, `vul_code`, `similar_codes` with `similar_code`/`similar_fix` pairs, and `similarity`, the cosine similarity of the most similar example rounded to one decimal). The examples were retrieved from the RAG database with OpenAI `text-embedding-3-small` embeddings and ChromaDB. The file also contains entries for VUL4J vulnerabilities outside the 42 evaluated ones; these are not used.

### `runs/`

Raw model outputs, organized as `runs/<model>/<setting>/<name>-run<k>.json` (k = 0-9). Each file is a JSON list with one object per vulnerability: `id`, `fixed_code` (the complete model response, including the reasoning text), and `error: true` if generation failed.

| Folder `<model>` | Model |
|---|---|
| `gpt20b` | `openai/gpt-oss-20b` |
| `gpt20b-safeguard` | `openai/gpt-oss-safeguard-20b` |
| `gpt120b` | `openai/gpt-oss-120b` |
| `gpt20b-cybernative` | `gpt-oss-20b` + LoRA adapter trained on CyberNative (r = 64, alpha = 128) |
| `gpt20b-megavul` | `gpt-oss-20b` + LoRA adapter trained on MegaVul (r = 16, alpha = 32) |

| Folder `<setting>` | Meaning | `--limit` |
|---|---|---|
| `base` | without RAG | - |
| `rag` | with RAG (default configuration in the paper) | `0` |
| `rag06` | with RAG and a similarity threshold of 0.6 | `0.6` |

### `results/`

One CSV file per configuration (`<model>.csv`, `<model>-rag.csv`, `<model>-rag06.csv`). Each row is one VUL4J vulnerability: `ID`, `CWE`, and `run0`-`run9`. Each run cell has the form `passed / executed`, the number of passed and executed tests after applying the generated patch. A run counts as a successful repair if all executed tests pass (`passed == executed > 0`); `0 / 0` means that no tests could be executed (e.g., the patch could not be applied or the project did not compile).

The metrics reported in the paper (pass@1, pass@10, Consistency) are computed from these files.

## Requirements

- A single NVIDIA GPU with 80 GB of memory (we used an H100)
- Python 3 with `torch`, `transformers`, `peft`, `trl`, `datasets`, `huggingface_hub`, `numpy`, and `vllm` (the latter only for `gpt_test_120b.py`)
- A Hugging Face access token

The scripts read the token from a `config.py` file in the working directory, which is not included:

```python
# config.py
login_token = "hf_..."
```

## Usage

### 1. Fine-tuning (`finetune_gpt.py`)

```bash
python finetune_gpt.py <run_name> <base_model> <data_path> <lora_r> <lora_alpha>

# MegaVul adapter
python finetune_gpt.py megavul openai/gpt-oss-20b datasets/megavul_vul_fixed_pairs_pairs.jsonl 16 32
# CyberNative adapter
python finetune_gpt.py cybernative openai/gpt-oss-20b datasets/cybernative.jsonl 64 128
```

The adapter is saved to `./final_adapter_<run_name>`.

Training setup: LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj` (dropout 0.1), BF16, 3 epochs, learning rate 3e-5 with cosine schedule and 10% warm-up, weight decay 0.1, batch size 1 with 16 gradient accumulation steps, and max gradient norm 0.5. 10% of the data (seed 42) is held out for evaluation every 10 steps; the checkpoint with the lowest evaluation loss is kept, with early stopping after 3 evaluations without improvement. Each training example is a chat with a system prompt, the vulnerable code as the user message, and the fixed code as the assistant message.

### 2. Inference with the 20B models (`gpt_test_rag.py`)

```bash
# Base model, without RAG
python gpt_test_rag.py --model-type base --model-name openai/gpt-oss-20b \
    --prompts prompts/prompts.json --output-name gpt20b-base --run 10

# Base model, with RAG
python gpt_test_rag.py --model-type base --model-name openai/gpt-oss-20b \
    --prompts prompts/prompts.json --rag prompts/rag.json --limit 0 \
    --output-name gpt20b-rag --run 10

# Fine-tuned model (loads ./final_adapter_<model-name>), with RAG and threshold 0.6
python gpt_test_rag.py --model-type finetuned --model-name megavul \
    --prompts prompts/prompts.json --rag prompts/rag.json --limit 0.6 \
    --output-name gpt20b-megavul-rag06 --run 10
```

For the safeguard model, use `--model-name openai/gpt-oss-safeguard-20b`. Outputs are written to `<output-name>-run<k>.json`.

Generation settings: Hugging Face `transformers`, native model precision, temperature 0.01, top-p 0.95, top-k 50, repetition penalty 1.05, at most 16,384 new tokens.

### 3. Inference with gpt-oss-120b (`gpt_test_120b.py`)

The model is loaded from a local directory:

```bash
huggingface-cli download openai/gpt-oss-120b --local-dir ./models/gpt-oss-120b

python gpt_test_120b.py --model-path ./models/gpt-oss-120b \
    --prompts prompts/prompts.json --rag prompts/rag.json --limit 0 \
    --output-name gpt-oss-120b-rag --run 10
```

Always pass `--model-path` explicitly: its default value points to `./models/gpt-oss-safeguard-120b`. Outputs are written to `vul4j_<output-name>-run<k>.json`.

Generation settings: vLLM, native MXFP4 weights, FP8 KV cache, maximum context of 8,192 tokens, temperature 0.01, top-p 0.9, at most 8,000 new tokens.

### RAG and the similarity threshold

With `--rag`, the retrieved examples of the vulnerability (vulnerable code and its fix) are appended to the prompt. With `--limit`, they are included only if the similarity of the most similar example is at least the given value; otherwise the prompt contains no examples.

### 4. Evaluation

The generated patches were evaluated with VUL4J: the code was extracted from each response, the original methods were replaced with the generated ones, and the project's tests, including the Proof-of-Vulnerability tests, were executed. The evaluation harness is not part of this package; its results are in `results/`.

