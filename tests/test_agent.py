#!/usr/bin/env python3.13
"""Test agent identity"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/Users/z/work/zen/zen-eco/agent/finetuned",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "/Users/z/work/zen/zen-eco/agent/finetuned",
    trust_remote_code=True
)

prompts = ["Who are you?", "What model are you?", "What are your capabilities?"]

for prompt in prompts:
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Q: {prompt}")
    print(f"A: {response.split('assistant')[-1].strip()}")
    print()