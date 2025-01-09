import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaTokenizer, LlamaForCausalLM
import torch

# model_name_or_id = "AI4Chem/ChemLLM-7B-Chat" 
# model_name_or_id = "AI4Chem/ChemLLM-20B-Chat-DPO"
# model_name_or_id = "AI4Chem/ChemLLM-20B-Chat-SFT"

model_name_or_id = "X-LANCE/ChemDFM-13B-v1.0"

# model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, trust_remote_code=True,
#                                              device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_id,trust_remote_code=True)

model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)
model.eval()
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=5,
        num_return_sequences=5,
        temperature=0.9,
        max_new_tokens=4,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True, 
    )

    outputs = model.generate(**inputs, generation_config=generation_config)
    generated_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    return [i[len(prompt):].strip() for i in generated_text]

prompt = \
    "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n\
    Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule. \
    The task is to predict the binary label for a given molecule, please answer with only 'Yes' or 'No', 'Unknown' is not allowed. \
    Please strictly follow the format, no other information can be provided.\n\
    SMILES: NS(=O)(=O)Cc1noc2ccccc12\n\
    Clinically-trail-toxic:"

    # "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n\
    # Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule. \
    # The task is to predict the binary label for a given molecule, please answer with only 'Yes' or 'No'. \
    # Please strictly follow the format, no other information can be provided. A template is provided in the beginning.\n\
    # SMILES: CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1\n\
    # Clinically-trail-toxic: Yes\n\
    # SMILES: NC(=O)C[S@@](=O)C(c1ccccc1)c1ccccc1\n\
    # Clinically-trail-toxic: No\n\
    # SMILES: NS(=O)(=O)Cc1noc2ccccc12\n\
    # Clinically-trail-toxic:"

    # SMILES: Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1\n\
    # Clinically-trail-toxic: Yes\n\
    # SMILES: Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl\n\
    # Clinically-trail-toxic: No\n\


    # "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n\
    # Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule. \
    # The task is to predict the binary label for a given molecule, please answer with only 'Yes' or 'No', 'Unknown' is not allowed. \
    # Please strictly follow the format, no other information can be provided.\n\
    # SMILES: NS(=O)(=O)Cc1noc2ccccc12\n\
    # Clinically-trail-toxic:"

    # "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n\
    # Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule. \
    # The task is to predict the binary label for a given molecule, please answer with only 'Yes' or 'No', 'Unknown' is not allowed. \
    # Please strictly follow the format, no other information can be provided. A template is provided in the beginning.\n\
    # SMILES: CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1\n\
    # Clinically-trail-toxic: Yes\n\
    # SMILES: NS(=O)(=O)Cc1noc2ccccc12\n\
    # Clinically-trail-toxic:"

    # SMILES: NC(=O)C[S@@](=O)C(c1ccccc1)c1ccccc1\n\
    # Clinically-trail-toxic: No\n\
    # SMILES: Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1\n\
    # Clinically-trail-toxic: Yes\n\
    # SMILES: Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl\n\
    # Clinically-trail-toxic: No\n\



response = generate_text(prompt)

print(response)