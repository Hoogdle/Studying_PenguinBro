import torch


t1 = torch.FloatTensor([[1,2,3,4],[1,5,4,6],[7,8,10,11]])
t2 = torch.LongTensor([1,2,1])

print(t1)
# tensor([[ 1.,  2.,  3.,  4.],
#         [ 1.,  5.,  4.,  6.],
#         [ 7.,  8., 10., 11.]])

print(t1.data)
# tensor([[ 1.,  2.,  3.,  4.],
#         [ 1.,  5.,  4.,  6.],
#         [ 7.,  8., 10., 11.]])

# t1과 t1.data의 출력을 동일하지만
# t1은 Tensor인 반면
# t1.data는 t1의 데이터만을 참조한다.