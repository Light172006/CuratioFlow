"""
Interactive inference script for the fine-tuned Qwen3-1.7B adapter.

Run with: python inference.py
Type your message and press Enter. Type 'end' to quit.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ------------------------------------------------------------------
# Must match the training script exactly — same base model, same
# quantization config, same adapter path.
# ------------------------------------------------------------------

model_name = "Qwen/Qwen3-1.7B"
adapter_path = "Qwen/Qwen3-1.7B-finetuned"   # matches `new_model` from your training script

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (quantized)...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()   # inference mode — disables dropout etc., which are only meant for training

# ------------------------------------------------------------------
# Single-turn generation function
# ------------------------------------------------------------------

def generate_response(user_input: str) -> str:
    messages = [{"role": "user", "content": user_input}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt portion so we only decode the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()

# ------------------------------------------------------------------
# Interactive loop
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\nModel loaded. Type your message, or 'end' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "end":
            print("Ending session.")
            break

        if not user_input:
            continue

        reply = generate_response(user_input)
        print(f"Model: {reply}\n")