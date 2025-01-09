import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertModel, BertTokenizerFast
from sklearn.metrics import accuracy_score,roc_auc_score
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pickle
# 读取文件
with open("datalst.pkl", "rb") as tf:
    datalst = pickle.load(tf)

model_name_or_id = "X-LANCE/ChemDFM-13B-v1.0"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)
model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")

# model_name_or_id = "AI4Chem/ChemLLM-20B-Chat-SFT"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_id,trust_remote_code=True)

import logging
logging.basicConfig(filename=f"{model_name_or_id.split('/')[-1]}_infer.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


answer=[]
label=[]
b=[]
a=[]
c=[]
d=[]
with torch.no_grad():
    for i in tqdm(range(len(datalst))):
        input_text = datalst[i]['instruction']
        input_text = f"Human: {input_text}\nAssistant:"
        logging.info(input_text)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        generation_config = GenerationConfig(
            do_sample=False,
            temperature=0.1,
            max_new_tokens=4,
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
            )
        outputs = model.generate(**inputs, generation_config=generation_config)
        generated_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0][len(input_text):]
        
        ans = generated_text.strip()
        if ans == 'Yes' or ans == 'yes':
            c.append(1)
        elif ans == 'No' or ans == 'no':
            c.append(0)
        else:
            c.append(random.choice([0,1]))

        answer.append(generated_text.strip())
        label.append(datalst[i]['output'])
        logits = outputs.scores
        probs = [torch.softmax(log, dim=-1) for log in logits]

        #print(torch.max(probs[0][0]), torch.argmax(probs[0][0]))
        #print(torch.max(probs[1][0]), torch.argmax(probs[1][0]))

        # ChemLLM
        # yesp = probs[0][0, 7560].item()
        # nop = probs[0][0, 2458].item()

        # ChemDFM
        yesp = probs[0][0, 3869].item()
        nop = probs[0][0, 1939].item()

        prob_max = tokenizer.decode(torch.argmax(probs[0][0]))
        logging.info([yesp, nop, prob_max, ans, datalst[i]['output']])
        sump = yesp + nop
        b.append(yesp / sump)
        d.append(nop / sump)

        if datalst[i]['output'] == 'Yes':
            a.append(1)
        elif datalst[i]['output'] == 'No':
            a.append(0)

logging.info(f"\n{'='*50}\n")
logging.info(answer)
logging.info(label)
logging.info([roc_auc_score(a, b)])
logging.info([accuracy_score(a, c)])
