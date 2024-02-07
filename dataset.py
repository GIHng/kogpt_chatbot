import numpy as np
import pandas as pd
import re
import torch
import urllib.request
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
import urllib.request
import os

filename = "ChatBotData.csv"
filepath = './' + filename

if not os.path.exists(filepath):
    print(f"{filename} 파일을 다운로드합니다.")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
        filename = filename,
    )

Chatbot_Data = pd.read_csv('ChatBotData.csv')
Chatbot_Data = Chatbot_Data[:300]

BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
SENT = "<unused1>"
Q_TKN = "<usr>"
A_TKN = "<sys>"


koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK,)

class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn["Q"]
        q = re.sub(r"([?.!,])", r" ", q)

        a = turn["A"]
        a = re.sub(r"([?.!,])", r" ", a)

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        if q_len > self.max_len:
            a_len = self.max_len - q_len

            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)) :]

                q_len = len(q_toked)
                a_len = self.max_len - q_len
            
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len

            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)):]

                q_len = len(q_toked)
                a_len = self.max_len - q_len
            
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        labels = [self.mask,] * q_len + a_toked[1:]

        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)

        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)

        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

            token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)

        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return (token_ids, np.array(mask), labels_ids)
    

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]

    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)


train_set = ChatbotDataset(Chatbot_Data, max_len=40)

train_dataloader = DataLoader(train_set, batch_size=32, num_workers=0,
                              shuffle=True, collate_fn=collate_batch,)

def get_train_dataloader(train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_batch):
    return DataLoader(train_set, batch_size, num_workers,shuffle, collate_fn)


print("start")
for batch_idx, samples in enumerate(train_dataloader):
    token_ids, mask, label = samples
    print("token_ids ====> ", token_ids)
    print("mask =====> ", mask)
    print("label =====> ", label)
print("end")


