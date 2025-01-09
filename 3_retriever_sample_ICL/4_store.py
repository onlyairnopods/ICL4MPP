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
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import deepchem
import deepchem.molnet
from rdkit import RDLogger
from rdkit import Chem
RDLogger.DisableLog('rdApp.*')

def canonicalize_smiles(smiles: str) -> str:
    """
    convert SMILES to Canonical SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        canonical_smiles = Chem.MolToSmiles(mol)
        return canonical_smiles
    except:
        return smiles

tasks, datasets, transformers = deepchem.molnet.load_clintox(splitter='scaffold', reload=True,
                                                             data_dir='../data/clintox_data',
                                                             save_dir='../data/clintox_datasets')

train_dataset, valid_dataset, test_dataset = datasets

train_dataset_df = train_dataset.to_dataframe()
test_dataset_df = test_dataset.to_dataframe()
train_dataset_df['ids'] = train_dataset_df['ids'].apply(lambda x: canonicalize_smiles(x))
test_dataset_df['ids'] = test_dataset_df['ids'].apply(lambda x: canonicalize_smiles(x))
train_input, train_label = train_dataset_df['ids'].values, train_dataset_df['y2'].values
test_input, test_label = test_dataset_df['ids'].values, test_dataset_df['y2'].values

train_label = ["Yes" if i == 1 else "No" for i in train_label]
test_label = ["Yes" if i == 1 else "No" for i in test_label]

device_x = "cuda:2"
device_p = "cuda:0"

# load model
model_name =  'unikei/bert-base-smiles'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model_x = BertModel.from_pretrained(model_name).to(device_x)
model_p = BertModel.from_pretrained(model_name).to(device_p)

model_x.load_state_dict(torch.load('/home/fangmiaoNLP/lzz/Chemllm/ICL/clintox_model_x-60.pt'))
model_p.load_state_dict(torch.load('/home/fangmiaoNLP/lzz/Chemllm/ICL/clintox_model_p-60.pt'))

model_x.to(device_x)
model_p.to(device_p)

MAX_LENGTH = 32
# Encode train data embedding
candidates_embed = {}
for i in tqdm(range(len(train_input))):
    with torch.no_grad():
        inputs = tokenizer(train_input[i], max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt').to(device_p)
        output = model_p(**inputs)
        candidates_embed[train_input[i]] = (F.normalize(output.pooler_output).to(device_x), train_label[i])

datalst = []
task_des_n='You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n' \
         'Please strictly follow the format, no other information can be provided. Given the SMlLES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is clinically trail toxic (Yes) or No clinically trail toxic (No) based on the SMILES string representation of each molecule. ' \
          'You will be provided with some molecule SMILES as examples, accompanied by a binary label indicating ' \
          'whether it is clinically trail toxic (Yes) or No clinically trail toxic (No) in the beginning.\n'

task_des_zero='You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n' \
         'Please strictly follow the format, no other information can be provided. Given the SMlLES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is clinically trail toxic (Yes) or No clinically trail toxic (No) based on the SMILES string representation of each molecule. ' \
          'If the molecule is clinically trail toxic, output Yes; otherwise output No. Please answer with only Yes or No.\n'

question='Then predict whether the following molecule it is clinically trail toxic or not. Please answer with only Yes or No.\n'

# Encode test data embedding
for i in tqdm(range(len(test_input))):
    with torch.no_grad():
        sim_lst = []
        inputs = tokenizer(test_input[i], max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt').to(device_x)
        output = model_x(**inputs)
        x_embed = F.normalize(output.pooler_output).to(device_x)

        for can, embed in candidates_embed.items():
            sim = torch.matmul(x_embed, embed[0].T)
            sim_lst.append([can, sim, embed[1]])

    sim_lst.sort(key=lambda x: x[1], reverse=True)

    k = 4
    candidates = sim_lst[:k]
    a = {}
    prompt = task_des_n
    for example in candidates:
        prompt += f"Input SMILES: {example[0]}\nLabel: {example[-1]}\n"
    prompt += question
    prompt += f"Input SMILES: {test_input[i]}\nAnswer: "
    a['instruction'] = prompt
    a['output'] = test_label[i]
    datalst.append(a)

import pickle
with open("datalst.pkl", "wb") as tf:
    pickle.dump(datalst,tf)
