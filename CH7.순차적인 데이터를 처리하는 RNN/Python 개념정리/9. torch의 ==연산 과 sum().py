import torch


t1 = torch.FloatTensor([[1],[2],[3],[4]])
t2 = torch.LongTensor([[1],[3],[3],[4]])
t3 = (t1==t2)

print(t3)
# tensor([[ True],
#         [False],
#         [ True],
#         [ True]])
print(t3.sum()) 
#tensor(3)

# == 연산으로 새로운 bool 텐서 생성가능
# .sum()으로 모든 요소의 합을 scalar 형태로 반환할 수 있다.