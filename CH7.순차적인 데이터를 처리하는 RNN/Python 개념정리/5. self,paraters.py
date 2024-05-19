import torch
import torch.nn as nn

model = nn.RNN(input_size=5,hidden_size=10,num_layers=1,batch_first=True)


it = model.parameters()

print(next(it).data.size())
print(next(it).data.size())

# torch.Size([10, 5])
# torch.Size([10, 10])

# self.paramter 의 반환 방식에 대해 알아봄.
# iter 형태로 반환을 하며 next를 통해 하나하나 출력해본 결과 Wxh, Whh의 가중치들을 차례차례 반환해줌.


