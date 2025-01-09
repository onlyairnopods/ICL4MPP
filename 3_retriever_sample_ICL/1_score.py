import deepchem
import deepchem.molnet
import pandas as pd
import random
import torch
import json
from tqdm import tqdm
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore")

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig,AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score,roc_auc_score

from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs,Torsions
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs,rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from rdkit import Chem
RDLogger.DisableLog('rdApp.*')

tasks, datasets, transformers = deepchem.molnet.load_clintox(splitter='scaffold', reload=True,
                                                             data_dir='../data/clintox_data',
                                                             save_dir='../data/clintox_datasets')

train_dataset, valid_dataset, test_dataset = datasets

train_dataset = valid_dataset

train_dataset_df = train_dataset.to_dataframe()
test_dataset_df = test_dataset.to_dataframe()
train_input, train_label = train_dataset_df['ids'].values, train_dataset_df['y2'].values
test_input, test_label = test_dataset_df['ids'].values, test_dataset_df['y2'].values

def top_k_maccs_similar_molecules(target_smiles, molecule_smiles_list, label_list, k = 5): # 从train set中选择target_smiles的KNN
    label_list = ["Yes" if i == 1 else "No" for i in label_list]
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_maccs = AllChem.GetMorganFingerprintAsBitVect(target_mol, 1, 1024)
    similarities = []
    for i, smiles in enumerate(molecule_smiles_list):
        if smiles == target_smiles:
            continue

        sample_maccs = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 1, 1024)
        # tanimoto_similarity = DataStructs.CosineSimilarity(target_maccs,sample_maccs)
        tanimoto_similarity = DataStructs.FingerprintSimilarity(target_maccs, sample_maccs, metric=DataStructs.TanimotoSimilarity)
        similarities.append((smiles, tanimoto_similarity, label_list[i]))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5_similar_molecules = similarities[:k]
    return top_5_similar_molecules

task_des_n='You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n' \
         'Please strictly follow the format, no other information can be provided. Given the SMlLES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is clinically trail toxic (Yes) or No clinically trail toxic (No) based on the SMILES string representation of each molecule. ' \
          'You will be provided with some molecule SMILES as examples, accompanied by a binary label indicating ' \
          'whether it is clinically trail toxic (Yes) or No clinically trail toxic (No) in the beginning.\n'

task_des_zero='You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n' \
         'Please strictly follow the format, no other information can be provided. Given the SMlLES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is clinically trail toxic (Yes) or No clinically trail toxic (No) based on the SMILES string representation of each molecule. ' \
          'If the molecule is clinically trail toxic, output Yes; otherwise output No. Please answer with only Yes or No.\n'

question='Then predict whether the following molecule it is clinically trail toxic or not. Please answer with only Yes or No.\n'

model_name_or_id = "AI4Chem/ChemLLM-20B-Chat-SFT"
model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_id, trust_remote_code=True)


dict_a = {}

for i in tqdm(range(len(train_input))): # 遍历train set
    lst = []
    query = train_input[i]
    scaffold_examples = top_k_maccs_similar_molecules(query, train_input, train_label, 4) # 选择train set中最近的64-NN
    for example in scaffold_examples:
        prompt = task_des_n
        prompt += f"Input SMILES: {example[0]}\nLabel: {example[-1]}\n"
        prompt += question
        prompt += f"Input SMILES: {query}\nAnswer: "
        input_text = f"Human: {prompt}\nAssistant:"

        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        generation_config = GenerationConfig(
            do_sample=False,
            temperature=0.1,
            max_new_tokens=4,
            repetition_penalty=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True, 
        )
        outputs = model.generate(**inputs, generation_config=generation_config)
        generated_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0][len(input_text):]
        ans = generated_text.strip()
        logits = outputs.scores
        probs = [torch.softmax(log, dim=-1) for log in logits]
        # print(torch.max(probs[0][0]), torch.argmax(probs[0][0]))
        # print(torch.max(probs[1][0]), torch.argmax(probs[1][0]))
        yesp = probs[0][0, 7560].item()
        nop = probs[0][0, 2458].item()
        sump = yesp + nop
        print(train_label[i], ans, yesp/sump, nop/sump)
        # 正确label的confidence作为score
        if train_label[i]==1:
            lst.append((example[0],example[-1], yesp/sump))
        elif train_label[i]==0:
            lst.append((example[0],example[-1], nop/sump))

    dict_a[query] = lst
    # print(lst)

info_json = json.dumps(dict_a, sort_keys=False, indent=4, separators=(',', ': '))
# 显示数据类型
with open('ClinTox_score.json', 'w') as f:
    f.write(info_json)