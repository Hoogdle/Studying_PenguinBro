### 자연어 전처리 ###

# '토치텍스트'의 전처리 도구들과 파이토치의 'nn.Embedding' 같은 기능으로 자연어 전치 과정을 쉽게 처리할 수 있다.

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

TEXT = data.Field(sequential=True, batch_first = True, lower=True)
LABEL = data.Field(sequential=False, batch_first = True)

trainset,testset = datasets.IMDB.splits(TEXT,LABEL)

TEXT.build_vocab(trainset,min_freq=5)
LABEL.build_vocab(trainset)

trainset, valset = trainset.split(split_ratio=0.8)
train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset,valset,testset),batch_size = BATCH_SIZE,shuffle=True,repeat=False)

vocab_size = len(TEXT.vocab)
n_classes = 2

print("[학습셋] : %d [검증셋] : %d [테스트셋] : %d [단어수] : %d [클래스] : %d" %(len(trainset),len(valset),len(testset),vocab_size,n_classes))