import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaTokenizer, LlamaForCausalLM

# 加载模型和tokenizer
model_name_or_id = "AI4Chem/ChemLLM-20B-Chat-SFT"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)
model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")

# 定义生成函数
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="CLI for ChemLLM-20B-Chat-SFT model")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the generated text")
    
    args = parser.parse_args()
    
    print("ChemLLM-20B-Chat-SFT CLI. Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        response = generate_text(prompt, max_length=args.max_length)
        print(f"Model: {response}")

if __name__ == "__main__":
    main()

