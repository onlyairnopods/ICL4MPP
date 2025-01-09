import deepchem
import deepchem.molnet
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from typing import Literal, Callable, Any, Tuple, List
import random
import numpy as np
import datetime
from rdkit import Chem, rdBase
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaTokenizer, LlamaForCausalLM
import argparse

rdBase.DisableLog('rdApp.warning')

def set_seed(seed):
    torch.manual_seed(seed)  # 设置CPU的随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python的随机种子
    torch.backends.cudnn.deterministic = True  # 确保cudnn的确定性
    torch.backends.cudnn.benchmark = False  # 确保cudnn的确定性

set_seed(1111)

tasks, datasets, transformers = deepchem.molnet.load_clintox(splitter='scaffold', reload=True,
                                                             data_dir='./data/clintox_data',
                                                             save_dir='./data/clintox_datasets')

train_dataset, valid_dataset, test_dataset = datasets

batch_size = 1
total_batches = len(test_dataset) // batch_size

class Model:
    def __init__(self, model_name_or_id: Literal["AI4Chem/ChemLLM-20B-Chat-SFT", "AI4Chem/ChemLLM-20B-Chat-DPO", "X-LANCE/ChemDFM-13B-v1.0"], **kwargs):
        assert model_name_or_id in ["AI4Chem/ChemLLM-20B-Chat-SFT", "AI4Chem/ChemLLM-20B-Chat-DPO", "X-LANCE/ChemDFM-13B-v1.0"], \
            "model must be one of 'AI4Chem/ChemLLM-20B-Chat-SFT', 'AI4Chem/ChemLLM-20B-Chat-DPO', 'X-LANCE/ChemDFM-13B-v1.0'"
        self.model_name_or_id = model_name_or_id

        self.yes_token_ids = [
            [7560,], # 7560为ChemLLM词表中Yes对应的id
            [3869,], # 3869为ChemDFM词表中Yes对应的id
            ]
        self.no_token_ids = [
            [2458, 2783],  # 2458为ChemLLM词表中No对应的id, 2783-Not
            [1939,],  # 1939为ChemDFM词表中No对应的id
            ]
        
        if "AI4Chem" in model_name_or_id:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_id, trust_remote_code=True)
            self.yes_token_id = self.yes_token_ids[0]
            self.no_token_id = self.no_token_ids[0]

        else: # ChemDFM        
            self.model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)
            self.yes_token_id = self.yes_token_ids[1]
            self.no_token_id = self.no_token_ids[1]

        self.generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            **kwargs,
            repetition_penalty=1.5,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

    def __call__(self, prompt: str, debug_mode: bool = False):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        input_ids = inputs['input_ids']
        outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        logits = outputs.scores
        generated_ids = outputs.sequences
        probs = [torch.softmax(log, dim=-1) for log in logits]

        output_token_ids = generated_ids[0][ len(input_ids[0]): ]
        response = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
        
        if debug_mode:
            for i, token_id in enumerate(output_token_ids):
                token_prob = probs[i][0, token_id].item()
                print(f"Token ID: {token_id}, Probability: {token_prob}")

        total_yesp, total_nop = 0., 0.
        for x in self.yes_token_id:
            total_yesp += probs[0][0, x].item()
        for y in self.no_token_id:
            total_nop += probs[0][0, y].item()
        
        sump = (total_yesp + total_nop) + 1e-14
        y_score = total_yesp / sump

        return [response], [y_score]

def smiles2maccs_fp(smiles: str):
    return MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))

def smiles2rdk_fp(smiles: str):
    return Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))

def smiles2morgan_fp(smiles: str):
    return AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), 2)

def calc_tanimoto_similarity(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def calc_cosine_similarity(fp1, fp2) -> float:
    return DataStructs.CosineSimilarity(fp1, fp2)   #####

def calc_dice_similarity(fp1, fp2) -> float:
    return DataStructs.DiceSimilarity(fp1, fp2)

class BasePrompter(object):
    def __init__(self, system_instruction: str = "", template: str = "", verbose: bool = False):
        self.system_instruction = system_instruction
        self.template = template
        self.verbose = verbose

    def generate_prompt(self, query_smiles):
        raise NotImplementedError
    
    def canonicalize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False)
    
class ZeroShotPrompter(BasePrompter):
    def __init__(self, system_instruction: str = "", template: str = "", verbose: bool = False):
        super().__init__(system_instruction, template, verbose)

    @staticmethod
    def generate_prompt(self, query_smiles):
        query_smiles = self.canonicalize_smiles(query_smiles)

        prompt = f"{self.template}\nSMILES:{query_smiles}\nAnswer:"

        if self.verbose:
            print(prompt)

        return prompt

class FewShotPrompter(BasePrompter):
    def __init__(self, system_instruction: str = "", template: str = "", verbose: bool = False, *, 
                 sample_dataset, sample_molecule_format: Literal['smiles', 'maccs_fp', 'rdk_fp', 'morgan_fp'], 
                 sample_mode: Literal['random', 'cosine_similarity', 'tanimoto_similarity', 'dice_similarity'], sample_num: int):
        super(FewShotPrompter, self).__init__(system_instruction, template, verbose)
        
        assert sample_mode in ['random', 'cosine_similarity', 'tanimoto_similarity', 'dice_similarity'], "mode must be either 'random' or 'cosine_similarity' or 'tanimoto_similarity', 'dice_similarity'\n"
        self.sample_mode = sample_mode
        assert sample_molecule_format in ['smiles', 'maccs_fp', 'rdk_fp', 'morgan_fp'], "mode must be either 'smiles' or 'maccs_fp' or 'rdk_fp' or 'morgan_fp'\n"
        self.sample_molecule_format = sample_molecule_format
        self.sample_num = sample_num
        self.sample_dataset = self.convert_molecule_format(sample_dataset)

    def convert_molecule_format(self, sample_dataset):
        sample_dataset = sample_dataset.to_dataframe()
        if self.sample_molecule_format == 'smiles':
            return sample_dataset
        else:
            sample_dataset[self.sample_molecule_format] = sample_dataset['ids'].apply(lambda x: eval(f'smiles2{self.sample_molecule_format}')(x))
            return sample_dataset
            
    def get_demonstrations(self, query_smiles: str) -> List:
        if self.sample_mode == 'random':
            return self.random_sample_examples(query_smiles)
        else:
            return self.similar_sample_examples(query_smiles, self.sample_num)

    def random_sample_examples(self, query_smiles: str) -> List[Tuple[str, str]]:
        # y2 = CT_TOX
        positive_examples = self.sample_dataset[self.sample_dataset["y2"] == 1].sample(int(self.sample_num/2))
        negative_examples = self.sample_dataset[self.sample_dataset["y2"] == 0].sample(int(self.sample_num/2))

        smiles = positive_examples["ids"].tolist() + negative_examples["ids"].tolist()
        smiles = [self.canonicalize_smiles(i) for i in smiles]
        class_label = positive_examples["y2"].tolist() + negative_examples["y2"].tolist()
        #convert 1 to "Yes" and 0 to "No"" in class_label
        class_label = ["Yes" if i == 1 else "No" for i in class_label]
        sample_examples = list(zip(smiles, class_label))
        return sample_examples

    def similar_sample_examples(self, query_smiles: str, top_k: int) -> List[Tuple[str, str]]:
        query_smiles = eval(f'smiles2{self.sample_molecule_format}')(query_smiles)
        similarities = []
        for k in self.sample_dataset[self.sample_molecule_format].tolist():
            similarities.append(eval(f'calc_{self.sample_mode}')(query_smiles, k))
        sample_idx = np.argsort(-np.array(similarities))[:top_k]

        smiles = []
        class_label = []
        for i in sample_idx:
            smiles.append(self.canonicalize_smiles(self.sample_dataset.iloc[i]['ids']))
            class_label.append(self.sample_dataset.iloc[i]['y2'])
        class_label = ["Yes" if i == 1 else "No" for i in class_label]
        sample_examples = list(zip(smiles, class_label))
        return sample_examples


    def generate_prompt(self, query_smiles: str) -> str:
        query_smiles = self.canonicalize_smiles(query_smiles)

        few_shot = ""
        demonstrations = self.get_demonstrations(query_smiles)
        for example in demonstrations:
            few_shot += f"SMILES:{example[0]}\nAnswer:{example[-1]}\n"

        prompt = f"{self.template}\n{few_shot}SMILES:{query_smiles}\nAnswer:"

        if self.verbose:
            print(prompt)
            
        return prompt
    
class FewShotPrompter1(FewShotPrompter):
    def generate_prompt(self, query_smiles):
        query_smiles = self.canonicalize_smiles(query_smiles)

        few_shot = ""
        demonstrations = self.get_demonstrations(query_smiles)
        for example in demonstrations:
            few_shot += f"SMILES:{example[0]}\nAnswer:{example[-1]}\n"

        prompt = f"{self.template}\n{few_shot}Is this molecule Clinically-trail-Toxic (Yes) or Not Clinically-trail-toxic (No)?\nSMILES:{query_smiles}\nAnswer:"

        if self.verbose:
            print(prompt)
            
        return prompt

def main(dataset: Any, 
         batch_size: int, 
         total_batches: int, 
         model: Callable[[str, bool], Tuple],
         prompt_generator: Callable[[str], str],
         ):
    y_trues = []
    y_scores = []
    responses = []
    cnt = 0
    
    for X, Y, W, ids in tqdm(dataset.iterbatches(batch_size=batch_size), total=total_batches):
        input_X = [prompt_generator(id) for id in ids]
        
        y_trues.extend(Y[:, -1])

        bs_responses, bs_y_scores = model(input_X)
        
        if cnt < 2:
            print(bs_responses, bs_y_scores)

        responses.extend(bs_responses)
        y_scores.extend(bs_y_scores)

        cnt += 1

        # if cnt > 3:
        #     break

    # print(responses)
    # print(y_trues)
    # print(y_scores)
    # print(cnt)

    roc = roc_auc_score(y_trues, y_scores)
    # print(roc)

    log = f"{responses}\n{y_trues}\n{y_scores}\n{cnt}\n{roc}"
    return log

#########################################################################################

zero_shot_prompt = \
    "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule. The task is to predict the binary label for a given molecule, please answer with only 'Yes' or 'No'."

few_shot_prompt = \
    "Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic (Yes) or Not Clinically-trail-toxic (No). The task is to predict the binary label for a given molecule, please answer with only 'Yes' or 'No'."

#########################################################################################

Models = ['AI4Chem/ChemLLM-20B-Chat-SFT', 'X-LANCE/ChemDFM-13B-v1.0']
Model_temperatures = [0.9]
ICL_sample_mode = ['cosine_similarity', 'tanimoto_similarity', 'dice_similarity']
ICL_sample_num = [2, 4, 8]
ICL_sample_molecule_format = ['maccs_fp', 'rdk_fp', 'morgan_fp']

log_folder = './log/'

params_idx = 0
params_idx_file = log_folder + 'params_idx.txt'

################################# test prompt
# few_shot_prompter = FewShotPrompter1(template=few_shot_prompt, sample_dataset=train_dataset, sample_molecule_format='maccs_fp', sample_mode='dice_similarity', sample_num=4)
# print(few_shot_prompter.generate_prompt('CCOC(=O)COc1cccc(NC(=O)c2cccc(-c3ccc(Cl)cc3)n2)c1'))

parser = argparse.ArgumentParser()
parser.add_argument('--params_idx', type=int, default=0)
args = parser.parse_args()

for temperature in Model_temperatures:
    for model in Models:
        # m = Model(model_name_or_id=model, temperature=temperature, max_new_tokens=10)
        
        for sample_mode in ICL_sample_mode:
            for sample_molecule_format in ICL_sample_molecule_format:
                for sample_num in ICL_sample_num:
                    
                    if params_idx < args.params_idx:
                        params_idx +=1
                        continue
                    m = None
                    del m
                    torch.cuda.empty_cache()
                    m = Model(model_name_or_id=model, temperature=temperature, max_new_tokens=10)

                    log_file = log_folder + f"{model.split('/')[-1]}_{temperature}_{sample_num}_{sample_molecule_format}_{sample_mode}.log"
                    now = datetime.datetime.now()
                    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_file, "w+") as file:
                        file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

                    few_shot_prompter = FewShotPrompter1(template=few_shot_prompt, sample_dataset=train_dataset, sample_molecule_format=sample_molecule_format, sample_mode=sample_mode, sample_num=sample_num)

                    log = main(dataset=test_dataset,
                            batch_size=batch_size,
                            total_batches=total_batches,
                            model=m,
                            prompt_generator=few_shot_prompter.generate_prompt)
                    
                    with open(log_file, "a") as file:
                        file.write(log + "\n")
                        file.write("=" * 50 + "\n")

                    params_idx += 1
                    with open(params_idx_file, 'w+') as f:
                        f.write(str(params_idx))
                    print(f"params_idx: {params_idx}")