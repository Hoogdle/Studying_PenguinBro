import torch


t1 = torch.FloatTensor([[1,2,3,4],[1,5,4,6],[7,8,10,11]])
t2 = torch.LongTensor([1,2,1])

print(t1)
# tensor([[ 1.,  2.,  3.,  4.],
#         [ 1.,  5.,  4.,  6.],
#         [ 7.,  8., 10., 11.]])
print(t1.max(0))
# torch.return_types.max(
# values=tensor([ 7.,  8., 10., 11.]),
# indices=tensor([2, 2, 2, 2]))
print(t1.max(1))
# torch.return_types.max(
# values=tensor([ 4.,  6., 11.]),
# indices=tensor([3, 3, 3]))


print('=================================')
print(t1.max(0)[0])
# tensor([ 7.,  8., 10., 11.])

print('=================================')
print(t1.max(0)[1])
# tensor([2, 2, 2, 2])

print('=================================')
temp = torch.FloatTensor([[1],[0],[1],[0]])
print(temp.size()) #torch.Size([4, 1])
print(t1.max(0)[0].view(temp.size()))
# tensor([[ 7.],
#         [ 8.],
#         [10.],
#         [11.]])

print('=================================')
print(t1.max(0)[1].size()) #torch.Size([4])
print(t1.max(0)[0].view(temp.size()).size()) #torch.Size([4, 1])



