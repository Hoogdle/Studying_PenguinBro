# 원본 코드 출처:
# https://github.com/keon/3-min-pytorch/blob/master/07-%E1%84%89%E1%85%AE%E1%86%AB%E1%84%8E%E1%85%A1%E1%84%8C%E1%85%A5%E1%86%A8%E1%84%8B%E1%85%B5%E1%86%AB_%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF_%E1%84%8E%E1%85%A5%E1%84%85%E1%85%B5%E1%84%92%E1%85%A1%E1%84%82%E1%85%B3%E1%86%AB_RNN/text_classification.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 40
USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False,batch_first=True)

trainset, testset = datasets.IMDB.splits(TEXT,LABEL)

TEXT.build_vocab(trainset,min_freq=5)
LABEL.build_vocab(testset)

trainset, valset = trainset.split(split_ratio=0.8)
train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset,valset,testset),batch_size=BATCH_SIZE,shuffle=True,repeat=False)

vocab_size = len(TEXT.vocab)
n_classes = 2

print("[TRAIN] : %d\t [TEST] : %d\t [VOCAB] : %d\t [CLASSES] : %d\n"%(len(trainset),len(valset),vocab_size,n_classes))
# [TRAIN] : 20000  [TEST] : 5000   [VOCAB] : 46159         [CLASSES] : 2

class BasicGRU(nn.Module):
    def __init__(self,n_layers,hidden_dim,n_vocab,embed_dim,n_classes,dropout_p=0.2):
        super(BasicGRU,self).__init__()
        print("Building Basic GRU Model")
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab,embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim,self.hidden_dim,num_layers=self.n_layers,batch_first=True)
        self.out = nn.Linear(self.hidden_dim,n_classes)

    def forward(self,x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size = x.size(0))
        x, _ = self.gru(x,h_0)
        h_t = x[:,-1,:]
        self.dropout(h_t)
        logit = self.out(h_t)
        return logit
    
    def _init_state(self,)