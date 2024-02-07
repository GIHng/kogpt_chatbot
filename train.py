import numpy as np
import pandas as pd
import torch
import urllib.request
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.module import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re
import os
from dataset import ChatbotDataset, collate_batch

filename = "ChatBotData.csv"
filepath = './' + filename

BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
SENT = "<unused1>"
Q_TKN = "<usr>"
A_TKN = "<sys>"

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                           bos_token = BOS, eos_token = EOS, unk_token='<unk>',
                                                           pad_token = PAD, mask_token = MASK)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

import urllib.request

if not os.path.exists(filepath):
    print(f"{filename} 파일을 다운로드합니다.")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
        filename = filename,
    )
Chatbot_Data = pd.read_csv("ChatbotData.csv")

device = torch.device("cpu")

train_set = ChatbotDataset(Chatbot_Data, max_len = 40)

train_dataloader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_batch)

model.to(device)
model.train()

learning_rate = 3e-5
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
Sneg = -1e18

print("=========== Train Start ===========")
for epoch in range(num_epochs):
    for batch_idx, samples in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids, mask, label = samples
        out = model(token_ids)
        out = out.logits

        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)

        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()

        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {avg_loss.item():.4f}")

    
print("=========== Train End ===========")

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        sent = '0'

        while 1:
            input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)

            pred = model(input_ids)
            pred = pred.logits
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]

            if gen == EOS:
                break
            a += gen.replace("_", " ")
        print("Chatbot > {}".format(a.strip()))