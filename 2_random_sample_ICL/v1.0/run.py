import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from model import build_model, set_seed
set_seed(1111)
from dataset import train_dataset, valid_dataset, test_dataset
from utils import load_cfg, input_format, save_result, radom_sample_examples, input_format_fs

# model_name_or_id = "AI4Chem/ChemLLM-7B-Chat" 
# model_name_or_id = "AI4Chem/ChemLLM-20B-Chat-DPO"
model_name_or_id = "X-LANCE/ChemDFM-13B-v1.0"
cfg_path = './info.yaml'
json_path = './info.json'
case = 'case6'
batch_size = 1
total_batches = len(test_dataset) // batch_size

model_cfg, prompts = load_cfg(cfg_path, json_path, case)

few_shot_nums = 2
few_shot_examples = radom_sample_examples(train_dataset, few_shot_nums)
# print(few_shot_examples)

predict, generate_text = build_model(model_name_or_id, model_cfg)

y_trues = []
y_scores = []
responses = []
cnt = 0
for X, Y, W, ids in tqdm(test_dataset.iterbatches(batch_size=batch_size), total=total_batches):
    # input_X = [input_format(*prompts, id) for id in ids]
    input_X = [input_format_fs(*prompts, few_shot_examples, id) for id in ids]
    # print(input_X)

    y_trues.extend(Y[:, -1])

    bs_responses, bs_y_scores = predict(input_X)
    print(bs_responses)

    # print(generate_text(input_X))
    responses.extend(bs_responses)
    y_scores.extend(bs_y_scores)

    cnt += 1

    # if cnt > 3:
    #     break

print(responses)
print(y_trues)
print(y_scores)
print(cnt)

roc = roc_auc_score(y_trues, y_scores)
print(roc)

save_result(cfg_path, json_path, case, responses, y_scores, roc)