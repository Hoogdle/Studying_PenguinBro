### RNN 모델 구현 ###

### BasicGRU라는 RNN 모델 구현 ###

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGRU(nn.Module):
    def __init__(self,n_layers,hidden_dim,n_vocab,embed_dim,n_classes,dropout_p=0.2):
        super(BasicGRU,self).__init__()
        print("Building Basic GRU model...")

        self.n_layers = n_layers # 은닉 벡터들의 '층' 갯수 정의
        self.embed = nn.Embedding(n_vocab,embed_dim) #n_vocab은 vocab의 사이즈, embed_dim은 임베딩 된 단어 텐서가 지니는 차원값
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim,self.hidden_dim,num_layers=self.n_layers,batch_first=True) #GRU는 LSTM과 마찬가지로 RNN에서의 기울기 소실 or 기울기 폭발 문제를 해결하는 메커니즘을 가지고 있다.

        # gru를 거친 문장들은 하나의 압축된 벡터로 변환된다.
        # 해당 벡터(정보)를 토대로 영화 리뷰가 긍정적인지 부정적인지 분류하기 위해 압축벡터를 아래와 같은 신경망을 통과하게 한다.
        self.out = nn.Linear(self.hidden_dim,n_classes)

    # h_0를 만들기 위한 함수, 초기에는 0으로 초기화 된다.
    def _init_state(self,batch_size=1):
        weight = next(self.parameters()).data # self.parameters()는 신경망 모듈(nn.Module)의 가중치 정보들을 반복자 형태로 반환
                                              # 해당 반복자가 생성하는 원소들은 실제 신경망의 가중치 텐서(.data)를 지닌 객체들
                                              # 즉 이 명령어는 nn.GRU 모델의 첫 번째 가중치 텐서를 추출
        return weight.new(self.n_layers,batch_size,self.hidden_dim).zero_() # 추출된 가중치 텐서를 (n_layers,batch_size,hidden_dim) 모양을 갖춘 텐서로 변환 후 텐서의 모든 값을 0으로 초기화

    def forward(self,x):
        x = self.embed(x) # 입력데이터 x(한 배치 속에 있는 모든 영화평)을 벡터의 배열로 전환
        h_0 = self._init_state(batch_size = x.size(0)) # 다른 신경망들과 달리 RNN은 입력 데이터 외에도 첫 번째 은닉 벡터 H0을 정의해 x와 함께 넣어줘야 한다.
        x, _ = self.gru(x,h_0) # self.gru의 결과값은 (batch_size,x의 length,hidden_dim) 모양을 가지는 3d 텐서
        h_t = x[:,-1,:] # 마지막 은닉벡터만 추출하여 h_t에 저장
        self.dropout(h_t)

        logit = self.out(h_t)
        return logit
    
    ### 학습함수와 평가함수

    def train(model,optimizer,train_iter):
        model.train() # 모델을 학습모드로 전환
        for b,batch in enumerate(train_iter): 
            x,y = batch.text.to(DEVICE), batch.label.to(DEVICE) # batch로 받은 train_iter의 text와 label 데이터를 x와y에 저장
            y.data.sub_(1) # 편의를 위해 라벨링 [1,2] -> [0,1]
            optimizer.zero_grad() # 기울기 누적을 피하기 위해 초기화
            logit = model(x) # 모델의 출력을 logit에, 모델의 출력은 실수 값이고 배치만큼의 차원을 가짐
            loss = F.cross_entropy(logit,y) # logit과 실제 값(y)와 cross entropy를 통해 손실값 계산
            loss.backward() # 역전파
            optimizer.step() # 가중치 갱신
