import json
import yaml
from rdkit import Chem

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False)

def get_instruction(json_path, key):
    with open(json_path, "r") as file:
        json_data = json.load(file)
    return json_data['system_instructions_pool'][key]

def get_prompt(json_path, key):
    with open(json_path, "r") as file:
        json_data = json.load(file)
    return json_data['prompts_pool'][key]

def load_cfg(cfg_path, json_path, case):
    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)
    select_prompt = cfg[case]['prompt']
    model_cfg = cfg[case]['model']
    system_instruction = cfg[case].get('system_instruction', None)

    instruction = get_instruction(json_path, key=system_instruction) if system_instruction else None
    prompt = get_prompt(json_path, key=select_prompt)

    return model_cfg, (instruction, prompt)

def save_result(cfg_path, json_path, case, responses, y_scores, roc):
    with open(cfg_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    case_yaml_content = yaml_data.get(case, {})

    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    if case not in json_data['cases_pool']:
        json_data['cases_pool'][case] = {}

    json_data['cases_pool'][case] = case_yaml_content
    json_data['cases_pool'][case]['responses'] = responses
    json_data['cases_pool'][case]['y_scores'] = y_scores
    json_data['cases_pool'][case]['ROC'] = roc

    with open(json_path, 'w') as file:
        json.dump(json_data, file, indent=4)


# from sklearn.metrics import accuracy_score,roc_auc_score
# a=[]
# b=[]
# outputs = model.generate(**inputs, generation_config=generation_config, output_scores=True,
#                          return_dict_in_generate=True, max_new_tokens=4)
# logits = outputs.scores
# probs = [torch.softmax(log, dim=-1) for log in logits]
# yesp = probs[0][0, 7560].item()#7560为ChemLLM词表中Yes对应的id
# nop = probs[0][0, 2458].item()#2458为ChemLLM词表中No对应的id
# sump = yesp + nop
# b.append(yesp / sump)
# if datalst[i]['output'] == 'Yes':
#     a.append(1)
# elif datalst[i]['output'] == 'No':
#     a.append(0)
# print(roc_auc_score(a, b))


def input_format(instruction, prompt, smiles=None):
    smiles = canonicalize_smiles(smiles)
    if instruction is not None:
        prefix_template=[
            "<|system|>:",
            "{}"
        ]
        prompt_template=[
            "<|user|>:",
            "{}\n",
            "SMILES: {}\n",
            "<|Bot|>:\n"
        ]
        system = f'{prefix_template[0]}\n{prefix_template[-1].format(instruction)}\n'
        prompt = f'\n{prompt_template[0]}\n{prompt_template[1].format(prompt)}{prompt_template[2].format(smiles)}{prompt_template[-1]}'

        return f"{system}{prompt}"

    else:
        prompt_template=[
            "{}\n",
            "SMILES: {}\n",
            "Clinically-trail-toxic: "
        ]
        prompt = f'\n{prompt_template[0].format(prompt)}\n{prompt_template[1].format(smiles)}{prompt_template[-1]}'
        return f"{prompt}"
    
def input_format_fs(instruction, prompt, few_shot_examples, smiles=None):
    smiles = canonicalize_smiles(smiles)
    if instruction is not None:
        prefix_template=[
            "<|system|>:",
            "{}"
        ]
        prompt_template=[
            "<|user|>:",
            "{}",
            "{}\n",
            "SMILES: {}\n",
            "<|Bot|>:\n"
        ]
        few_shot = ""
        for example in few_shot_examples:
            few_shot += f"\nSMILES: {example[0]}\n<|Bot|>: {example[-1]}"
        system = f'{prefix_template[0]}\n{prefix_template[-1].format(instruction)}\n'
        prompt = f'\n{prompt_template[0]}\n{prompt_template[1].format(prompt)}\n{prompt_template[2].format(few_shot)}{prompt_template[3].format(smiles)}{prompt_template[-1]}'

        return f"{system}{prompt}"

    else:
        prompt_template=[
            "{}",
            "{}\n",
            "SMILES: {}\n",
            "Clinically-trail-toxic: "
        ]
        few_shot = ""
        for example in few_shot_examples:
            few_shot += f"\nSMILES: {example[0]}\nClinically-trail-toxic: {example[-1]}"
        prompt = f'\n{prompt_template[0].format(prompt)}\n{prompt_template[1].format(few_shot)}{prompt_template[2].format(smiles)}{prompt_template[-1]}'
        return f"{prompt}"


def radom_sample_examples(datasets, sample_size):
    datasets = datasets.to_dataframe()
    # y2 = CT_TOX
    positive_examples = datasets[datasets["y2"] == 1].sample(int(sample_size/2))
    negative_examples = datasets[datasets["y2"] == 0].sample(int(sample_size/2))

    smiles = positive_examples["ids"].tolist() + negative_examples["ids"].tolist()
    smiles = [canonicalize_smiles(i) for i in smiles]
    class_label = positive_examples["y2"].tolist() + negative_examples["y2"].tolist()
    #convert 1 to "Yes" and 0 to "No"" in class_label
    class_label = ["Yes" if i == 1 else "No" for i in class_label]
    bace_examples = list(zip(smiles, class_label))
    return bace_examples


if __name__ == "__main__":
    print(canonicalize_smiles('c1ccc2c(c1)c(no2)CS(=O)(=O)N'))