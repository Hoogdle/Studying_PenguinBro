import torch
import torch.nn as nn


class Mymode(nn.Module):
    def __init__(self):
        super(Mymode,self).__init__()
        self.one = nn.Linear(5,4,bias=False)
        self.two = nn.Linear(4,2,bias=False)



model = Mymode()

it = model.parameters()
print(next(it).data)
print(next(it).data)

# model.paramters()는 반복자 객체를 반환
# next로 반복하다보면 다음 모델진행의 가중치를 반환해줌

# tensor([[ 0.1256, -0.1109, -0.4101, -0.2200,  0.1726],
#         [-0.1219,  0.0323,  0.3166, -0.0096, -0.4167],
#         [ 0.1113,  0.2748,  0.2156,  0.4271,  0.0945],
#         [ 0.4209, -0.2999, -0.1951, -0.4387, -0.0492]])
# tensor([[-0.2956, -0.2868,  0.2960, -0.1916],
#         [ 0.2009, -0.1696, -0.0402,  0.0553]])