### 자연어 전처리 ###

# '토치텍스트'의 전처리 도구들과 파이토치의 'nn.Embedding' 같은 기능으로 자연어 전치 과정을 쉽게 처리할 수 있다.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets # torchtext : 자연어 데이터셋을 다루기 위한 라이브러리


# 모델의 하이퍼파라미터 정의
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 40 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# 데이터셋을 설정
TEXT = data.Field(sequential=True, batch_first = True, lower=True)
LABEL = data.Field(sequential=False, batch_first = True)

# trainset과 testset으로 데이터 나눠주기
trainset,testset = datasets.IMDB.splits(TEXT,LABEL)

# 단어 사전 만들기
TEXT.build_vocab(trainset,min_freq=5) # 5번 이상 등장하는 단어만 사전에 담기, 5번 미만은 unk(Unknown) 토큰으로 대체
LABEL.build_vocab(trainset)

trainset, valset = trainset.split(split_ratio=0.8) # 학습셋 80%, 검증셋 20%로 나누기
# train_iter,val_iter,test_iter에 각각 배치 데이터 생성해주기
train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset,valset,testset),batch_size = BATCH_SIZE,shuffle=True,repeat=False)

# 사전속 단어 갯수와 레이블 수를 변수에 저장
vocab_size = len(TEXT.vocab)
n_classes = 2

print("[학습셋] : %d [검증셋] : %d [테스트셋] : %d [단어수] : %d [클래스] : %d" %(len(trainset),len(valset),len(testset),vocab_size,n_classes))
# [학습셋] : 20000 [검증셋] : 5000 [테스트셋] : 25000 [단어수] : 46159 [클래스] : 2