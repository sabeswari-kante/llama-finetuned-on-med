

import json
import random
# FAST INSTALL (Pre-compiled)
# Run this ONLY if the previous install is taking forever
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" accelerate bitsandbytes
# Force install the latest PEFT to fix the "ensure_weight_tying" error
!pip install --upgrade --force-reinstall "peft @ git+https://github.com/huggingface/peft.git"

# 1. Force-align Torch and Torchvision
# We strictly request the version Unsloth is asking for (>=0.24.0)
!pip install --upgrade "torch==2.9.1" "torchvision>=0.24.0" "torchaudio>=2.9.0"

# 2. Re-install Unsloth (Just to be safe after the torch update)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

import os
# Force Single GPU Mode
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
# 1. LOAD MODEL
max_seq_length = 2048 
print("⬇️ Loading Llama-3.1-8B (Senior Resident)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 2. CONFIGURE ADAPTERS
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. DEFINE ROBUST FORMATTING FUNCTION (The Fix)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    # Case 1: Batch Mode (List of strings) - The Trainer uses this during training
    if isinstance(examples["instruction"], list):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
            texts.append(text)
        return texts # <--- Returns a LIST (Correct for Trainer)

    # Case 2: Single Example Mode - Unsloth uses this for the safety check
    else:
        instruction = examples["instruction"]
        input       = examples["input"]
        output      = examples["output"]
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        return [text] # <--- Returns a LIST of 1 string

print("📂 Loading 'FINAL_MASTER_DATASET.json'...")
dataset = load_dataset("json", data_files="FINAL_MASTER_DATASET.json", split="train")

# Split into Train/Test
dataset = dataset.train_test_split(test_size=0.05) 

print(f"📊 Final Training on {len(dataset['train'])} cases")
print(f"📊 Evaluating on {len(dataset['test'])} cases")

# 4. TRAINING ARGUMENTS
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    
    # --- Pass the Function Here (No manual mapping needed) ---
    formatting_func = formatting_prompts_func, 
    
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 120,        
        learning_rate = 2e-5,   
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "medical_llama_final_v1",
        
        # Validation Settings
        eval_strategy = "steps",
        eval_steps = 10,
        save_strategy = "steps",
        save_steps = 20,
    ),
)

print("🚀 Starting Final Training...")

trainer.train()
print("✅ FINAL MODEL READY.")