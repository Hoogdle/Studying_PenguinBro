#!/usr/bin/env python
# coding: utf-8

# # Seq2Seq 기계 번역
# 이번 프로젝트에선 임의로 Seq2Seq 모델을 아주 간단화 시켰습니다.
# 한 언어로 된 문장을 다른 언어로 된 문장으로 번역하는 덩치가 큰 모델이 아닌
# 영어 알파벳 문자열("hello")을 스페인어 알파벳 문자열("hola")로 번역하는 Mini Seq2Seq 모델을 같이 구현해 보겠습니다.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


vocab_size = 256  # 총 아스키 코드 개수
x_ = list(map(ord, "hello"))  # 문자를 10진수의 유니코드의 리스트로 변환
y_ = list(map(ord, "hola"))   # 문자를 10진수의 유니코드의 리스트로 변환
print("hello -> ", x_) #hello ->  [104, 101, 108, 108, 111]
print("hola  -> ", y_) #hola  ->  [104, 111, 108, 97]


x = torch.LongTensor(x_) # 리스트를 Tensor로 변환해준다.
y = torch.LongTensor(y_)

print(x) #tensor([104, 101, 108, 108, 111])
print(y) #tensor([104, 111, 108,  97])


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.n_layers = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size) #vocab안의 단어의 갯수 만큼 벡터를 만들어야 하고 각 단어 벡터의 차원을 hidden_size로(이번 task만 특별히 원래는 단어 dim이 따로 존재) 해야하기 때문에 인자 주기
        self.encoder = nn.GRU(hidden_size, hidden_size) #batch_first = True를 하지 않았기 때문에 embedding = [seq_len, batch_size, embedding_size] 의 모양이다.
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.project = nn.Linear(hidden_size, vocab_size) # hidden 차원에서 vocab의 갯수만큼의 차원으로의 MLP

    def forward(self, inputs, targets):
        # 인코더에 들어갈 입력
        initial_state = self._init_state()
        embedding = self.embedding(inputs).unsqueeze(1) # (5,hidden_size) 였던 Tensor를 (5,1,hiddens_size) Tensor로 바꿔준다. (batch_size가 1이므로 상관 x)
                                                        # initial_state가 3차원이므로 차원 맞춰주기
        
        # 인코더 (Encoder)
        encoder_output, encoder_state = self.encoder(embedding, initial_state) # encoder_state는 마지막 은닉 상태 벡터를 받게된다.

        # 디코더에 들어갈 입력
        decoder_state = encoder_state # 마지막 은닉상태 벡터 즉, 문장의 총 의미를 담고 있는 벡터를 decoder_state에 넣어준다.
        decoder_input = torch.LongTensor([0]) # decoder의 초기 input은 0으로 설정, 디코더의 문장의 시작을 알리기 위함이며 공백문자 Null을 뜻한다.
        
        # 디코더 (Decoder)
        outputs = []
        
        for i in range(targets.size()[0]): # targets의 모양의 0번째 인덱스 크기만큼, 즉 여기에서는 halo에서 h,a,l,o의 갯수만큼 반복
            decoder_input = self.embedding(decoder_input).unsqueeze(1) # (1,hidden_size) 였던 Tensor를 (1,1,hiddens_size) Tensor로 바꿔준다. (batch_size가 1이므로 상관 x)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state) # decoder_input과 이전의 decoder_state를 인자로 넣어준다.
                                                                                       # decoder_output은 (1,1,hidden_size)의 모양
            projection = self.project(decoder_output) # projection의 결과로 (1,1,vocab_size)의 모양의 텐서가 나온다.
            outputs.append(projection) # 위 결과의 텐서를 outputs에 넣어준다.
            
            #티처 포싱(Teacher Forcing) 사용
            decoder_input = torch.LongTensor([targets[i]]) # 학습의 효율을 위해 진짜 target을 다음의 input으로 넣어준다.

        outputs = torch.stack(outputs).squeeze() # outputs에는 각 시계열의 rnn output 값들이 저장되어있다. torch.stack으로 하나의 텐서로 모아준 후 요소가 1인 차원을 모두 삭제한다
                                                 # 결과로 (4,256)의 텐서가 반환된다.
        return outputs
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()


seq2seq = Seq2Seq(vocab_size, 16)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)


log = []
for i in range(1000): # 앞의 과정과 동일
    prediction = seq2seq(x, y)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_val = loss.data
    log.append(loss_val)
    if i % 100 == 0:
        print("\n 반복:%d 오차: %s" % (i, loss_val.item()))
        _, top1 = prediction.data.topk(1, 1)
        print([chr(c) for c in top1.squeeze().numpy().tolist()])


plt.plot(log)
plt.ylabel('cross entropy loss')
plt.show()
