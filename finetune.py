import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_name = "Qwen/Qwen3-1.7B"
 
new_model = "Qwen/Qwen3-1.7B-finetuned"
output_dir = "./results"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # nf4 = normal-float 4bit, better than fp4 for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # Ada Lovelace (RTX 40-series) supports bf16 natively
    bnb_4bit_use_double_quant=True,     # quantizes the quantization constants too — extra memory savings
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",       # let accelerate place layers on your single GPU
)
model.config.use_cache = False   # incompatible with gradient checkpointing, must disable during training

peft_config = LoraConfig(
    r=32,                     # rank of the adapter matrices — higher = more capacity, more VRAM
    lora_alpha=64,            # scaling factor, commonly set to 2x r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[          # Qwen3's attention + MLP projection layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

import pandas as pd
dataset = pd.read_csv(r'D:\CuratioFlow\data\data.csv')

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,     # start small — raise only if VRAM allows
    gradient_accumulation_steps=4,     # simulates a batch size of 2*4=8 without extra VRAM
    gradient_checkpointing=True,       # trades compute time for VRAM — essential on 6GB
    optim="paged_adamw_8bit",          # paged optimizer avoids VRAM spikes on gradient updates
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    weight_decay=0.001,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                         # matches compute_dtype above
    dataset_text_field="text",
    packing=False,
    report_to="none",
    eval_strategy="steps",
    eval_steps=100,
)

def format_example(example):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,)

from datasets import Dataset

dataset = pd.read_csv(r'D:\CuratioFlow\data\data.csv')   # your existing pandas load
dataset = Dataset.from_pandas(dataset)                    # convert here
dataset = dataset.train_test_split(test_size=0.1, seed=42)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],          # your raw dataset, unformatted
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    formatting_func=format_example,  # just the function reference here
    processing_class=tokenizer,
    args=training_args,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(new_model)
    print(f"Saved fine-tuned adapter to {new_model}")