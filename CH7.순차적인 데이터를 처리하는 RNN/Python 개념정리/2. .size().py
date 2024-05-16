# pytorch에서 .size()연산에 관한 정리

import torch

a = torch.FloatTensor([[1,2],[3,4],[4,2]])
print(a.size()) #torch.Size([3, 2])
print(a.size(0)) #3 
# .size(num)은 0번째(1번째) 차원의 크기를 알려줌