import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

vocab_size = 256 # 총 아스키 코드의 갯수

x_ = list(map(ord,"hello")) # ord는 문자를 아스키코드롤 변환 해주는 함수 map으로 각각의 문자열 마다 적용하여 list화 한다.
y_ = list(map(ord,"hola"))
print("hello -> ",x_) #hello ->  [104, 101, 108, 108, 111]
print("halo -> ",y_) #halo ->  [104, 111, 108, 97]
x = torch.LongTensor(x_)
y= torch.LongTensor(y_)
print(x) #tensor([104, 101, 108, 108, 111])
print(y) #tensor([104, 111, 108,  97])

### 텐서 공부를 위한 주석
# print(x.shape) #torch.Size([5])
# print(y.shape) #torch.Size([4])
# print(x.unsqueeze(1).shape) #torch.Size([5, 1])

class Seq2Seq(nn.Module):
    def __init__(self,vocab_size,hidden_size):
        super(Seq2Seq,self).__init__()
        self.n_layers = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.encoder = nn.GRU(hidden_size,hidden_size)
        self.decoder = nn.GRU(hidden_size,hidden_size)
        self.project = nn.Linear(hidden_size,vocab_size)

    def forward(self,inputs,targets):
        initial_state = self._init_state()
        embedding = self.embedding(inputs).unsqueeze(1) # 1차원 -> 2차원
        encoder_output, encoder_state = self.encoder(embedding,initial_state)
        decoder_state = encoder_state
        decoder_input = torch.LongTensor([0])

        outputs = []
        
        for i in range(targets.size()[0]):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input,decoder_state)

            # 디코더의 출력값으로 다음 글자를 예측
            porjection = self.project(decoder_output)
            outputs.append(porjection)

            # 티처 포싱을 위해 디코더의 입력을 갱신한다.
            decoder_input = torch.LongTensor([targets[i]])
        outputs = torch.stack(outputs).squeeze() # squeeze는 차원이 1인 차원을 제거해준다. ex) [4,1] -> [4]
        return outputs
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data # next는 반복이 끝나더라도 StopIteration이 발생하지 않고 기본값을 출력해준다.
        # self.parameters는 model의 paramaters를 반복해주는
        return weight.new(self.n_layers,batch_size,self.hidden_size).zero_()


seq2seq = Seq2Seq(vocab_size,16)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq.parameters(),lr=1e-3)

log = []

for i in range(1000):
    prediction = seq2seq(x,y)
    loss = criterion(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_val = loss.data
    log.append(loss_val)
    if i%100==0:
        print("\n반복 : %d 오차 :%s" %(i,loss_val.item()))
        _,top1 = prediction.data.topk(1,1)
        print([chr(c) for c in top1.squeeze().numpy().tolist()])

plt.plot(log)
plt.ylabel('cross entropy loss')
plt.show()

# 반복 : 0 오차 :5.536358833312988
# ['b', '³', '1', 'L']

# 반복 : 100 오차 :1.7992615699768066
# ['h', 'o', 'l', 'a']

# 반복 : 200 오차 :0.6917393207550049
# ['h', 'o', 'l', 'a']

# 반복 : 300 오차 :0.44230353832244873
# ['h', 'o', 'l', 'a']

# 반복 : 400 오차 :0.2769744098186493
# ['h', 'o', 'l', 'a']

# 반복 : 500 오차 :0.1639757752418518
# ['h', 'o', 'l', 'a']

# 반복 : 600 오차 :0.11047646403312683
# ['h', 'o', 'l', 'a']

# 반복 : 700 오차 :0.08037871867418289
# ['h', 'o', 'l', 'a']

# 반복 : 800 오차 :0.06158099323511124
# ['h', 'o', 'l', 'a']

# 반복 : 900 오차 :0.049031805247068405
# ['h', 'o', 'l', 'a']
