import torch
import torch.nn as nn

temp = torch.FloatTensor([[[1,2,3,4]]])
print(temp)
print(temp.size())

model = nn.Linear(4,2,bias=False)

result = model(temp)

print(result)
print(result.size())


# Linear를 거친 결과로 
# torch.Size([1, 1, 4]) => torch.Size([1, 1, 2]) 로 변환되었다
# 즉 nn.Linear의 인자는 마지막 차원의 요소에 맞춰줘야 한다.