import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaTokenizer, LlamaForCausalLM
import numpy as np
import random

def build_model(model_name_or_id, model_cfg):
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, trust_remote_code=True,
    #                                          device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_id, trust_remote_code=True)

    model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, trust_remote_code=True,
                                             device_map="auto")
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id, trust_remote_code=True)

    top_k = model_cfg['top_k']
    temperature = model_cfg['temperature']
    max_new_tokens = model_cfg['max_new_tokens']
    repetition_penalty = model_cfg['repetition_penalty']

    # def generate_output(prompt):
    #     inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    #     generation_config = GenerationConfig(
    #         do_sample=True,
    #         top_k=1,
    #         temperature=0.9,
    #         max_new_tokens=500,
    #         repetition_penalty=1.5,
    #         pad_token_id=tokenizer.eos_token_id,
    #         output_scores=True, 
    #         return_dict_in_generate=True
    #     )

    #     input_ids = inputs['input_ids']
    #     outputs = model.generate(**inputs, generation_config=generation_config)
    #     logits = outputs.scores
    #     generated_ids = outputs.sequences
    #     probs = [torch.softmax(log, dim=-1) for log in logits]

    #     output_token_ids = generated_ids[0][ len(input_ids[0]): ]
    #     response = tokenizer.decode(output_token_ids, skip_special_tokens=True)

    #     yesp = probs[0][0, 7560].item() # 7560为ChemLLM词表中Yes对应的id
    #     nop = probs[0][0, 2458].item() # 2458为ChemLLM词表中No对应的id
    #     sump = (yesp + nop) + 1e-14
    #     y_score = yesp / sump

    #     return response, y_score

    def generate_text(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.9,
            max_new_tokens=1,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id
        )

        outputs = model.generate(**inputs, generation_config=generation_config)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def predict(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True, 
            return_dict_in_generate=True
        )

        input_ids = inputs['input_ids']
        outputs = model.generate(**inputs, generation_config=generation_config)
        logits = outputs.scores
        generated_ids = outputs.sequences
        probs = [torch.softmax(log, dim=-1) for log in logits]
        
        bs = len(input_ids)

        responses = []
        y_scores = []
        for i in range(bs):
            output_token_ids = generated_ids[i][ len(input_ids[i]): ]
            responses.append(tokenizer.decode(output_token_ids, skip_special_tokens=True))
        
            for i, token_id in enumerate(output_token_ids):
                token_prob = probs[i][0, token_id].item()
                print(f"Token ID: {token_id}, Probability: {token_prob}")

            # yesp = probs[i][-1, 9583].item() # 7560为ChemLLM词表中Yes对应的id
            # nop = probs[i][-1, 2917].item() # 2458为ChemLLM词表中No对应的id

            yesp = probs[i][-1, 3869].item() # 7560为ChemLLM词表中Yes对应的id
            nop = probs[i][-1, 1939].item() # 2458为ChemLLM词表中No对应的id
            
            # print(yesp, nop, '-----')
            sump = (yesp + nop) + 1e-14
            y_scores.append(yesp / sump)

        return responses, y_scores

    return predict, generate_text


def set_seed(seed):
    torch.manual_seed(seed)  # 设置CPU的随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python的随机种子
    torch.backends.cudnn.deterministic = True  # 确保cudnn的确定性
    torch.backends.cudnn.benchmark = False  # 确保cudnn的确定性