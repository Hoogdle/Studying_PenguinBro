import torch
import torch.nn as nn
import numpy as np
output = torch.Tensor([[12,21,18]])
result = nn.LogSoftmax(dim=1)
value = result(output)
print(value)
target = torch.LongTensor([2])
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
print(loss) 

# tensor([[-9.0487, -0.0487, -3.0487]])
# tensor(3.0487)

# nn.crossentropy는 nn.logsoftmax를 포함한다.
# nn.logsoftmax는 softmax의 결과에 log를 취하는 것이고 
# log를 취하는이유는 crossentropy 과정(식)에 log가 존재하기 때문이다.
# 이후 target의 인덱스를 정하여 부호만 바꿔주면 완성!
