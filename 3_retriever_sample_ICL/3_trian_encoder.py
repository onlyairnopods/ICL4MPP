import pandas as pd
import json
import random
import numpy as np
from tqdm import tqdm
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BertTokenizerFast, BertModel

from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, rdBase
rdBase.DisableLog('rdApp.debug')
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')

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

# init seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)  # 设置CPU的随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python的随机种子

set_seed(0)

# load model
model_name =  'unikei/bert-base-smiles'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model_x = BertModel.from_pretrained(model_name).to(device)
model_p = BertModel.from_pretrained(model_name).to(device)

# load optimizer
optimizer_x = optim.AdamW(model_x.parameters(), lr=5e-5)
optimizer_p = optim.AdamW(model_p.parameters(), lr=5e-5)

# load dataset
class MyDataset(Dataset):
    def __init__(self, json_path, pos_neg_num=8):
        super().__init__()
        self.data = json.load(open(json_path, 'r'))
        self.data_keys = list(self.data.keys())
        self.K = pos_neg_num

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, index):
        anchor = self.data_keys[index]

        positives = self.data[anchor][:self.K]
        negatives = self.data[anchor][-self.K:]
        pos = random.choice(positives)[0]
        neg = random.choice(negatives)[0]
        anchor = canonicalize_smiles(anchor)
        pos = canonicalize_smiles(pos)
        neg = canonicalize_smiles(neg)
        return anchor, pos, neg


# training
EPOCHS=60 #30
MAX_LENGTH=40 # 你可以根据需要设置这个长度
BATCH_SIZE=74
mydataset = MyDataset('./ClinTox_score_sort.json', pos_neg_num=2)
dataloader = DataLoader(mydataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
# torch.autograd.set_detect_anomaly(True)

losses = []
for epoch in tqdm(range(EPOCHS)):
    for step, batch in enumerate(dataloader):
        optimizer_x.zero_grad()
        optimizer_p.zero_grad()
        
        anchors, b_pos, b_neg = batch
        # anchors: List[str], len == bs
        # b_pos: List[str], len == bs
        # b_neg: List[str], len == bs
        
        anchors_input = tokenizer(
            anchors,
            max_length=MAX_LENGTH,  
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)

        pos_input = tokenizer(
            b_pos,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        neg_input = tokenizer(
            b_neg,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt' 
        ).to(device)

        anchor_embeds = F.normalize(
            model_x(**anchors_input).pooler_output
         ) # (bs, dim=768)
        pos_embeds = F.normalize(
            model_p(**pos_input).pooler_output
        ) # (bs, dim=768)
        neg_embeds = F.normalize(
            model_p(**neg_input).pooler_output
        ) # (bs, dim=768)

        
        # pos_neg_embeds.T # (dim=768, 2bs), [:][0:bs-1]为pos, [:][bs-1:2bs-1为neg
        # pos_neg_embeds:
            # p0 * 768
            # p1 * 768
            # p2 * 768
            # p3 * 768
            # n0 * 768
            # n1 * 768
            # n2 * 768
            # n3 * 768
        # pos_neg_embeds.T:
            # p0    p1    p2    p3    n0    n1    n2    n3
            #  *    *      *    *     *      *     *    *
            # 768   768   768  768    768   768   768  768
        pos_neg_embeds = torch.cat([pos_embeds, neg_embeds], dim=0) # (2bs, dim=768)    [0:bs-1]为pos, [bs-1:2bs-1]为neg
        scores = anchor_embeds @ pos_neg_embeds.T # (bs, 2bs) 其中[0,0], [1,1], ..., [b-1, b-1]是真正的正样本
        # # loss = -( log(e^sim) - log(e^sim + sum(e^sim)) ) = -( sim - log(e^sim + sum(e^sim))
        loss = -torch.log(
            torch.exp(torch.diag(scores, 0)) / 
            torch.sum(
                torch.exp(scores), dim=1,
            )
        ) # (b,)
        loss = loss.mean()  # tensor(1.2842)

        #### 公式版+SimCLR版
        # mask = torch.eye(BATCH_SIZE, dtype=torch.float32).to(device)
        # features = torch.cat([pos_embeds.unsqueeze(1), neg_embeds.unsqueeze(1)], dim=0)
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # == pos_neg_embeds
        # contrast_count = features.shape[1]
        # anchor_count= 1
        # anchor_feature = anchor_embeds
        # logits = torch.matmul(anchor_feature, contrast_feature.T)    # == scores
        # mask_ = mask.repeat(anchor_count, contrast_count)
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(BATCH_SIZE * anchor_count).view(-1, 1).to(device),
        #     0
        # )
        # _mask = mask_ * logits_mask
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mask_pos_pairs = _mask.sum(1)
        # mask_pos_pairs_ = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        # mean_log_prob_pos = (_mask * log_prob).sum(1) / mask_pos_pairs_
        # loss = (-mean_log_prob_pos).view(anchor_count, BATCH_SIZE).mean()   # tensor(1.0661)

        ### MoCo版
        # l_pos = torch.matmul(anchor_embeds, pos_embeds.T)
        # l_neg = torch.matmul(anchor_embeds, neg_embeds.T)
        # logits = torch.cat([l_pos, l_neg], dim=1)   # (2, 4) tensor([[0.9570, 0.8721, 0.6338, 0.6479], [0.9126, 0.9272, 0.8594, 0.8696]]
        # labels = torch.arange(len(anchor_embeds), dtype=torch.long, device=logits.device)
        # loss_my = F.cross_entropy(logits, labels, reduction='mean')    # tensor(1.2842)

        ### CSDN
        # query = anchor_embeds
        # positive_logit = torch.sum(query * pos_embeds, dim=1, keepdim=True) # (N, 1) == torch.diag(scores, 0).view(2,1) tensor([[0.9570], [0.9272]]
        # query = anchor_embeds.unsqueeze(1)  # (N, 1, D)
        # negative_logits = query @ neg_embeds.T # (N, 1, M)
        # negative_logits = negative_logits.squeeze(1)  # (N, M) == l_neg
        # logits = torch.cat([positive_logit, negative_logits], dim=1)  # (N, 1+M) == (2, 3) tensor([[0.9570, 0.6338, 0.6479],[0.9272, 0.8594, 0.8696]] 
        # labels = torch.arange(len(query), dtype=torch.long, device=query.device)  # (N,)
        # loss = F.cross_entropy(logits, labels, reduction='mean')

        loss.backward()
        optimizer_x.step()
        optimizer_p.step()

        print(f'epoch: {epoch}, step: {step}, loss: {loss.item():.4f}')
        losses.append(loss.item())

    if (epoch + 1) == EPOCHS:
        torch.save(model_x.state_dict(), f'./clintox_model_x-{epoch+1}.pt')
        torch.save(model_p.state_dict(), f'./clintox_model_p-{epoch+1}.pt')

plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('./clintox_loss_curve.png')