import os, sys, gc, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, EarlyStoppingCallback
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login
import config


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def main():
    if len(sys.argv) < 3:
        print("Usage: python gpt_oss_20b_lora_sft.py <run_name> <model_name_or_path> [data_path]")
        sys.exit(1)

    run_name = sys.argv[1]
    model_name = sys.argv[2]
    data_path = sys.argv[3] if len(sys.argv) > 3 else getattr(config, "data_path", None)
    if not data_path:
        print("ERROR: data_path not provided and config.data_path missing.")
        sys.exit(2)

    login(token=config.login_token)

    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f"Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(
        model_name,
        token=config.login_token,
        trust_remote_code=True,
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = getattr(tok, "pad_token", None) or getattr(tok, "eos_token", None)

    print(f"Loading dataset from: {data_path}")
    ds_all = load_dataset("json", data_files={"data": data_path})["data"]

    def _ok(ex):
        return bool(ex.get("code_vul")) and bool(ex.get("code_sec"))
    ds_all = ds_all.filter(_ok)

    ds_split = ds_all.train_test_split(test_size=0.1, seed=42)
    ds_train = ds_split["train"]
    ds_eval = ds_split["test"]

    print(f"Dataset size: train={len(ds_train)}, eval={len(ds_eval)} (from total={len(ds_all)})")

    cleanup()

    print("Loading GPT-OSS 20B model (BF16, device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=config.login_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",          
        low_cpu_mem_usage=True,
        use_cache=False,            
    )

    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()

  
    lora_cfg = LoraConfig(
    r=int(sys.argv[4]),                          
    lora_alpha=int(sys.argv[5]),                 
    lora_dropout=0.1,             
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  
    bias="none",
    task_type="CAUSAL_LM",    )

    cleanup()

  
    max_len = 4096


    eval_steps = 10

    args = SFTConfig(
        output_dir=f"./final_adapter_{run_name}",
        num_train_epochs=3,                   
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,        
        learning_rate=3e-5,                    
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        max_grad_norm=0.5,
        report_to="none",                        
        gradient_checkpointing=True,
        eval_accumulation_steps=1,
        packing=False,
    )

    def format_chat(example):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior Java developer specializing in identifying and patching software "
                    "vulnerabilities in production code. Produce a complete, final, production-ready patch."
                ),
            },
            {
                "role": "user",
                "content": f"Patch this vulnerable Java code:\n\n```java\n{example['code_vul']}\n```",
            },
            {
                "role": "assistant",
                "content": f"Patched code:\n```java\n{example['code_sec']}\n```",
            },
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        formatting_func=format_chat,
        peft_config=lora_cfg,  
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],

    )

    try:
        trainer.model.print_trainable_parameters()
    except Exception:
        pass

    print("Training")
    trainer.train()

    print("Saving best adapter + tokenizer")
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()

 