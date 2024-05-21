import torch

t1 = torch.FloatTensor([[1,2,3]])
print(t1.size()) #torch.Size([1, 3])

t1 = t1.squeeze(dim=1) # 2번째 차원의 요소가 1이 아니므로 삭제 불가
print(t1.size()) #torch.Size([1, 3])

t1 = t1.squeeze()
print(t1.size()) #torch.Size([3])

t1 = t1.unsqueeze(dim=0)
print(t1.size()) #torch.Size([1, 3])


# .squeeze는 차원 중 요소가 1인 차원을 삭제하는 함수
# .unsqueeze는 요소가 1인 차원을, 즉 차원을 증가시켜주는 함수

# .squeeze의 arg중 dim은 해당 차원의 요소가 1이면 해당 차원을 삭제한다.
# .unsqueeze는 dim에 대한 arg를 필요로 하며 해당 arg의 dim을 늘려준다.


temp = torch.FloatTensor([[[1,2,3,4]]])
print(temp)
print(temp.size())

# tensor([[[1., 2., 3., 4.]]])
# torch.Size([1, 1, 4])

temp = temp.squeeze()
print(temp)
print(temp.size())

# tensor([1., 2., 3., 4.])
# torch.Size([4])


# squeeze를 하면 요소가 1인 차원은 '모두' 삭제된다!