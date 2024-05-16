#!/usr/bin/env python
# coding: utf-8


# 출처 : https://github.com/keon/3-min-pytorch/blob/master/07-%E1%84%89%E1%85%AE%E1%86%AB%E1%84%8E%E1%85%A1%E1%84%8C%E1%85%A5%E1%86%A8%E1%84%8B%E1%85%B5%E1%86%AB_%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF_%E1%84%8E%E1%85%A5%E1%84%85%E1%85%B5%E1%84%92%E1%85%A1%E1%84%82%E1%85%B3%E1%86%AB_RNN/text_classification.py

# # 프로젝트 1. 영화 리뷰 감정 분석
# **RNN 을 이용해 IMDB 데이터를 가지고 텍스트 감정분석을 해 봅시다.**
# 이번 책에서 처음으로 접하는 텍스트 형태의 데이터셋인 IMDB 데이터셋은 50,000건의 영화 리뷰로 이루어져 있습니다.
# 각 리뷰는 다수의 영어 문장들로 이루어져 있으며, 평점이 7점 이상의 긍정적인 영화 리뷰는 2로, 평점이 4점 이하인 부정적인 영화 리뷰는 1로 레이블링 되어 있습니다. 영화 리뷰 텍스트를 RNN 에 입력시켜 영화평의 전체 내용을 압축하고, 이렇게 압축된 리뷰가 긍정적인지 부정적인지 판단해주는 간단한 분류 모델을 만드는 것이 이번 프로젝트의 목표입니다.


# 2024-05-15 미해결
# cross entropy 에서 label는 1과 0이 어떻게 처리 되는가
# 어떻게 vocab의 인덱스들을 자동으로 가져올 수 있는가 .text와 .label ??
# h0 정리
# gru output 정리
# eval 정리

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets


# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", DEVICE)


# 데이터 로딩하기
print("데이터 로딩중...")
TEXT = data.Field(sequential=True, batch_first=True, lower=True) # sequential true => 토큰나이징 O , batch_first => 첫 번째 입력차원을 batch 크기만큼, 
LABEL = data.Field(sequential=False, batch_first=True)  # lower => 컴퓨터는 "apple"과 "Apple"을 다른 단어로 인식
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)  # trainset과 testset에는 각각 text field와 label field가 존재하게 된다.
                                                      # trainset[0] {'text' : ['if','you','like',~~~~~~~,'picture.'],'label' : 'pos}

TEXT.build_vocab(trainset, min_freq=5) # data.Feild의 build_vocab함수로 최소 5번 이상 나온 단어들을 vocab에 포함. 5번 미만은 <unk> 토큰으로 인덱싱 (단어를 인덱싱한 vocab을 생성, TEXT.vocab에 저장)
LABEL.build_vocab(trainset) 

# 학습용 데이터를 학습셋 80% 검증셋 20% 로 나누기
trainset, valset = trainset.split(split_ratio=0.8) # trainset을 다시 trainset과 valset으로 8:2 비율로 나눠주기
train_iter, val_iter, test_iter = data.BucketIterator.splits( # data.BucketIterator로 반복자를 batch의 크기를 한 단위로 하여 세팅해줌                                                            
        (trainset, valset, testset), batch_size=BATCH_SIZE,   # BucketIterator는 비슷한 길이를 가지는 문장을 동일한 배치에 포함시키는 기능을 가짐 => padding을 minimize 할 수 있다.                                                              
        shuffle=True, repeat=False)                           # BucketIterator는 단어들을 vocab의 인덱싱 넘버로 변경해주기도 함. (word -> indexed number) 
                                                              # shuffle = True => 매 epoch 마다 shuffle 해줌, 무작위로 섞어줘서 데이터 순서를 학습하지 않도록, 일반화 성능을 높여줌.

vocab_size = len(TEXT.vocab)
n_classes = 2


print("[학습셋]: %d [검증셋]: %d [테스트셋]: %d [단어수]: %d [클래스] %d"
      % (len(trainset),len(valset), len(testset), vocab_size, n_classes))


class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicGRU, self).__init__() # BasicGRU의 부모 클래스를 상속
        print("Building Basic GRU model...")
        self.n_layers = n_layers # GRU의 layers의 수, layer가 깊을 수록 복잡한 연산 수행 가능
        self.embed = nn.Embedding(n_vocab, embed_dim) # nn.Embedding은 input: indexed된 단어 list, output : 해당 단어의 벡터를 만들어준다.
                                                      # embedding된 단어, 즉 indexed list => vector list로 변환된 텐서가 GRU의 input으로 들어가게 됩니다.
        self.hidden_dim = hidden_dim 
        self.dropout = nn.Dropout(dropout_p) # overfitting을 막기 위해 dropout 설정
        self.gru = nn.GRU(embed_dim, self.hidden_dim, 
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes) # 마지막 hidden_state를 MLP를 거쳐 n_classes(==2)의 차원으로 변환, 일종의 분류기. 이후 cross entropy 진행

    def forward(self, x):
        x = self.embed(x) # indexed된 단어가 embedding의 input으로 => indexed_num => word's vector 로 변환됨.
        h_0 = self._init_state(batch_size=x.size(0)) # batch_size를 텐서 x의 첫 번째 차원. 즉, 배치 크기만큼으로 설정. h_0(초기 은닉 상태)는 (layer의 갯수,배치크기,은닉 차원)의 크기의 텐서로 0으로 채워지게 된다.
        x, _ = self.gru(x, h_0)  # gru의 input으로 embeded된 단어의 텐서, h_0 텐서가 들어가고 output으로 (배치사이즈,문자길이,은닉차원)의 텐서와 , h_t 텐서를 출력, 이중 첫 번째 텐서만 받음
        h_t = x[:,-1,:] # 마지막 단어 input일 때의 hidden state만 추출하여 h_t에 저장
        self.dropout(h_t)
        logit = self.out(h_t)  # [b, h] -> [b, o]
        return logit
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()


def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()


def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[이폭: %d] 검증 오차:%5.2f | 검증 정확도:%5.2f" % (e, val_loss, val_accuracy))
    
    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss


model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))
