import torch
import torch.nn as nn

input = torch.FloatTensor([[1,2,3,4],[2,3,4,5],[6,2,1,5]])
print(input.size())
target = torch.FloatTensor([[0],[2],[1]])

model = nn.GRU(num_layers=1,hidden_size=3,input_size=4,batch_first=True)

result, last= model(input)
print(result)
# tensor([[0.8725, 0.5374, 0.4805],
#         [0.9947, 0.7017, 0.7674],
#         [0.9996, 0.9853, 0.8651]], grad_fn=<SqueezeBackward1>)
print(last)
# tensor([[0.9996, 0.9853, 0.8651]], grad_fn=<SqueezeBackward1>)

# last는 마지막 은닉 상태 벡터 값을 받게 된다.