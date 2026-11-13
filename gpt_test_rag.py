import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import config
import json
import argparse


def cleanup_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def load_base_gpt_oss_model(model_name):
    print(f"Loading base GPT-OSS model: {model_name}")
    cleanup_memory()

    try:
        model_loading_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "token": config.login_token,
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True
        }

        print("Loading GPT-OSS base model (native precision)...")
        print("Expected memory: ~16GB (MXFP4) or ~48GB (bfloat16)")

        model = AutoModelForCausalLM.from_pretrained(**model_loading_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.login_token)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token for GPT-OSS")

        print_memory_usage()
        print("Base GPT-OSS model and tokenizer loaded successfully")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading base GPT-OSS model: {e}")
        cleanup_memory()
        return None, None


def load_finetuned_gpt_oss_model(adapter_path):
    print(f"Loading GPT-OSS adapter: {adapter_path}")
    cleanup_memory()

    try:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = peft_config.base_model_name_or_path

        print(f"Base model: {base_model_name}")

        model_loading_kwargs = {
            "pretrained_model_name_or_path": base_model_name,
            "token": config.login_token,
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True
        }

        print("Loading GPT-OSS base model (native precision)...")
        print("Expected memory: ~16GB (MXFP4) or ~48GB (bfloat16)")

        base_model = AutoModelForCausalLM.from_pretrained(**model_loading_kwargs)

        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("LoRA adapter loaded successfully")

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token for GPT-OSS")

        print_memory_usage()
        print("GPT-OSS model and tokenizer loaded successfully")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading GPT-OSS model: {e}")
        cleanup_memory()
        return None, None


def format_vulnerability_prompt_for_gpt_oss(vulnerable_code, rag, similar_codes):
    messages = [
        {
            "role": "system",
            "content": "You are a senior Java developer specializing in identifying and patching software vulnerabilities in critical code for production environments. Your current task is to analyze and patch vulnerabilities in a provided Java code snippet. Remember: Finding and patching the vulnerability is essential for project success. Your patch must be complete, final, and production-ready."
        },
        {
            "role": "user",
            "content": vulnerable_code + ( format_similar_codes(similar_codes) if rag else "" ) + f"\nReply with only the code inside java block.\n"
        }
    ]
    print("---PROMPT---")
    print(messages)
    print("------------")

    return messages

def format_similar_codes(similar_codes):
    if similar_codes == None:
        return ""
    f = "\nHere are some similar vulnerable code examples and their fixes.\n"
    for similar_code in similar_codes:
        f += f"Similar vulnerable code:\n```java\n{similar_code['similar_code']}\n```\n Patched code:\n```java\n{similar_code['similar_fix']}\n```\n"
    return f

def generate_response_gpt_oss(model, tokenizer, messages, max_new_tokens=16384, temperature=0.01):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=16384
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            top_p=0.95,
            top_k=50,
            use_cache=True,
        )

    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )
    return response.strip()


def fix_vulnerable_code_gpt_oss(model, tokenizer, vulnerable_code, args, vul_id):

    messages = format_vulnerability_prompt_for_gpt_oss(vulnerable_code, args.rag is not None, find_similar_code(args, vul_id))

    response = generate_response_gpt_oss(
        model,
        tokenizer,
        messages,
        max_new_tokens=16384,
        temperature=0.01
    )
    return response


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def find_similar_code(args, vul_id):
    if args.rag is not None:
        rag = load_data(args.rag)
        for r in rag:
          if r["id"] == vul_id and r["similarity"][0] >= float(args.limit):
            return r["similar_codes"]
    return None


def main():
    parser = argparse.ArgumentParser(description='GPT-OSS Vulnerability Testing')
    parser.add_argument('--model-type', choices=['base', 'finetuned'], required=True,
                        help='Model type: base (plain model) or finetuned (LoRA adapter)')
    parser.add_argument('--model-name', required=True,
                        help='Model name or adapter path')
    parser.add_argument('--output-name', required=True,
                        help='Output file name')
    parser.add_argument('--prompts', required=True,
                        help='Prompt file name')
    parser.add_argument('--rag',
                        help='RAG file name')
    parser.add_argument('--run', default=1,
                        help='Number of runs')
    parser.add_argument('--limit', default=0,
                        help='Similarity threshold for RAG')

    args = parser.parse_args()

    print(f"Model type: {args.model_type}")
    print(f"Model name/path: {args.model_name}")

    if args.model_type == 'base':
        print("Loading base GPT-OSS model...")
        model, tokenizer = load_base_gpt_oss_model(args.model_name)
    else:  # finetuned
        print("Loading finetuned GPT-OSS model...")
        model, tokenizer = load_finetuned_gpt_oss_model(f"final_adapter_{args.model_name}")

    if model is None:
        print("Failed to load GPT-OSS model!")
        return

    print("Loading test data...")
    vul4j = load_data(args.prompts)
    total_vulnerabilities = len(vul4j)
    for run in range(int(args.run)):
        print(f"----RUN{run}----")
        result = []
        for i, vul in enumerate(vul4j, 1):
           print(f"\n{'=' * 60}")
           print(f"Processing {i}/{total_vulnerabilities}: {vul['id']}")
           print("=" * 60)
           try:
              fixed_code = fix_vulnerable_code_gpt_oss(model, tokenizer, vul["prompt"], args, vul['id'])
              print(f"Fixed Code:\n{fixed_code}")
              result.append({
                      "id": vul["id"],
                      "fixed_code": fixed_code
                     })
           except Exception as e:
              print(f"Error processing {vul['id']}: {e}")
              result.append({
                            "id": vul["id"],
                            "fixed_code": f"ERROR: {str(e)}",
                            "error": True
                         })

              print("=" * 60)
        output_file = f"{args.output_name}-run{run}.json"
        save_data(output_file, result)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()