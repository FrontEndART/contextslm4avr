from pathlib import Path
from typing import List, Dict
import json
import argparse
import traceback

import torch
from vllm import LLM, SamplingParams


MODEL_PATH = Path("./models/gpt-oss-120b")
MAX_MODEL_LEN = 8192              
KV_DTYPE = "fp8"                   
TP_SIZE = 1                       
GPU_UTIL = 0.95                   

def load_data(prompts_file: str) -> List[Dict]:

    print(f"Loading test data from: {prompts_file}")

    with open(prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} vulnerabilities")
    return data


def save_data(output_file: str, data: List[Dict]):

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_file}")

def load_model(
    model_path: Path = MODEL_PATH,
    enable_lora: bool = False,
    lora_path: str = None,
    max_lora_rank: int = 64
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Download:\n"
            "huggingface-cli download openai/gpt-oss-120b --local-dir ./models/gpt-oss-120b"
        )

    print("Mode vLLM...")
    print(f"Model: {model_path}")
    print(f"MXFP4 natív")
    print(f"FP8 KV-cache")
    print(f"Max context: {MAX_MODEL_LEN} tokens")

    if enable_lora:
        print(f"LoRA adapter: ENABLED")
        if lora_path:
            print(f"LoRA path: {lora_path}")
        print(f"Max LoRA rank: {max_lora_rank}")

    vllm_config = {
        "model": str(model_path),
        "dtype": "auto",
        "tensor_parallel_size": TP_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "kv_cache_dtype": KV_DTYPE,
        "gpu_memory_utilization": GPU_UTIL,
        "trust_remote_code": True,
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }

    if enable_lora:
        vllm_config.update({
            "enable_lora": True,
            "max_lora_rank": max_lora_rank,
        })

    llm = LLM(**vllm_config)

    print("Model OK!")
    print_vram()

    return llm


def chat_to_harmony(messages: List[Dict[str, str]]) -> str:
    parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()

        if role not in ("system", "user", "assistant"):
            role = "user"

        parts.append(f"<|{role}|>\n{content}\n")

    if not messages or messages[-1].get("role") != "assistant":
        parts.append("<|assistant|>\n")

    return "".join(parts)


def chat(llm: LLM, messages: List[Dict[str, str]], max_tokens: int = 8000) -> str:
    prompt = chat_to_harmony(messages)

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.01,
        top_p=0.9,
        repetition_penalty=1.0,
    )
    output = llm.generate([prompt], sampling)[0]

    return output.outputs[0].text


def fix_vulnerable_code_gpt_oss(llm: LLM, prompt: str, rag, similar_codes, max_tokens: int = 8000) -> str:
    system_prompt = "You are a senior Java developer specializing in identifying and patching software vulnerabilities in critical code for production environments. Your current task is to analyze and patch vulnerabilities in a provided Java code snippet. Remember: Finding and patching the vulnerability is essential for project success. Your patch must be complete, final, and production-ready."


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt + ( format_similar_codes(similar_codes) if rag else "" ) + f"\nReply with only the code inside java block.\n"

    }
    ]
    print(messages)
    fixed_code = chat(llm, messages, max_tokens=max_tokens)

    return fixed_code.strip()

def format_similar_codes(similar_codes):
    if similar_codes == None:
        return ""
    f = "\nHere are some similar vulnerable code examples and their fixes.\n"
    for similar_code in similar_codes:
        f += f"Similar vulnerable code:\n```java\n{similar_code['similar_code']}\n```\n Patched code:\n```java\n{similar_code['similar_fix']}\n```\n"
    return f


def print_vram():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"VRAM: {allocated:.2f} GB (peak: {peak:.2f} GB / {total:.0f} GB)")


def find_similar_code(args, vul_id):
    if args.rag is not None:
        rag = load_data(args.rag)
        for r in rag:
          if r["id"] == vul_id and r["similarity"][0] >= float(args.limit):
            return r["similar_codes"]
    return None



def process_vulnerabilities(args):
    print("\n" + "="*80)
    print("GPT-OSS-120B Vulnerability Fixing")
    print("="*80)

    llm = load_model(MODEL_PATH)

    print("\nLoading test data...")
    vul4j = load_data(args.prompts)
    total_vulnerabilities = len(vul4j)
    for run in range(int(args.run)):
        print(f"----RUN{run}----")
        result = []
        for i, vul in enumerate(vul4j, 1):
            print(f"\n{'='*60}")
            print(f"Processing {i}/{total_vulnerabilities}: {vul['id']}")
            print("="*60)

            try:
                fixed_code = fix_vulnerable_code_gpt_oss(
                    llm,
                    vul["prompt"],
                    args.rag is not None,
                    find_similar_code(args, vul['id']),
                    max_tokens=args.max_tokens
                )

                preview = fixed_code[:200] if len(fixed_code) > 200 else fixed_code
                print(f"Fixed Code Preview:\n{preview}...")

                result.append({
                    "id": vul["id"],
                    "fixed_code": fixed_code
                })

            except Exception as e:
                print(f"Error processing {vul['id']}: {e}")
                traceback.print_exc()

                result.append({
                    "id": vul["id"],
                    "fixed_code": f"ERROR: {str(e)}",
                    "error": True
                })

            print("="*60)

            if i % 10 == 0:
                print_vram()

        output_file = f"vul4j_{args.output_name}-run{run}.json"
        save_data(output_file, result)

        print(f"\n{'='*80}")
        print(f"DONE!")
        print(f"{len(result)} vulnerabilities")
        print(f"OK: {len([r for r in result if not r.get('error')])}")
        print(f"ERROR: {len([r for r in result if r.get('error')])}")
        print(f"Results: {output_file}")
        print("="*80)

        print_vram()


def parse_args():

    parser = argparse.ArgumentParser(
        description="GPT-OSS-120B Vulnerability Fixing with vLLM"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to vulnerability prompts JSON file"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="results",
        help="Output file name prefix (default: results)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Maximum tokens to generate per fix (default: 8000)"
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Runs (default: 1)"
    )
    parser.add_argument(
        "--rag",
        type=str,
        help="Rag file name"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/gpt-oss-safeguard-120b",
        help="Path to GPT-OSS model (default: ./models/gpt-oss-120b)"
    )
    parser.add_argument('--limit', default=0,
                        help='Rag limit')


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model_path:
        MODEL_PATH = Path(args.model_path)

    process_vulnerabilities(args)

