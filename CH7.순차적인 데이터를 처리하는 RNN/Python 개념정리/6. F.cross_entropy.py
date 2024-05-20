# import torch

# temp = torch.FloatTensor([1])
# print(temp.item())


import torch
import torch.nn.functional as F


t1 = torch.FloatTensor([[1,2,3,4],[1,5,4,6],[7,8,10,11]])
t2 = torch.LongTensor([1,2,1])


result = F.cross_entropy(input=t1,target=t2,reduction='none')
print(result) #tensor([2.4402, 2.4121, 3.3618])

result = F.cross_entropy(input=t1,target=t2,reduction='sum')
print(result) #tensor(8.2141)

result = F.cross_entropy(input=t1,target=t2)
print(result) #tensor(2.7380)

