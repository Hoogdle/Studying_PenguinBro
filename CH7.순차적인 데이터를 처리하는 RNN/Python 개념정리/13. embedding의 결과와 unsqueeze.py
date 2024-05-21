import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


vocab_size = 256  # 총 아스키 코드 개수
x_ = list(map(ord, "hello"))  # 문자를 10진수의 유니코드의 리스트로 변환
y_ = list(map(ord, "hola"))   # 문자를 10진수의 유니코드의 리스트로 변환
print("hello -> ", x_) #hello ->  [104, 101, 108, 108, 111]
print("hola  -> ", y_) #hola  ->  [104, 111, 108, 97]


x = torch.LongTensor(x_) # 리스트를 Tensor로 변환해준다.
y = torch.LongTensor(y_)

print(x) #tensor([104, 101, 108, 108, 111])
print(y) #tensor([104, 111, 108,  97])

embedding = nn.Embedding(256,4)

print(embedding(x))

# tensor([[-0.5447,  0.0195, -1.5320, -0.0382],
#         [ 0.7517,  1.1877, -1.1136, -0.5459],
#         [ 0.0898, -0.4211,  0.8033,  0.3175],
#         [ 0.0898, -0.4211,  0.8033,  0.3175],
#         [ 1.2902, -1.0310,  1.1614,  0.6054]], grad_fn=<EmbeddingBackward0>)

print(embedding(x).unsqueeze(1))

# tensor([[[ 0.3719, -0.8734, -0.9587, -0.1095]],

#         [[ 1.4786, -1.3498,  0.6729, -0.9692]],

#         [[ 0.6428,  0.5456,  0.7366, -0.9442]],

#         [[ 0.6428,  0.5456,  0.7366, -0.9442]],

#         [[ 1.0812, -0.6853,  0.1137,  2.3680]]], grad_fn=<UnsqueezeBackward0>)

# embedding의 결과와 unsqueeze의 결과를 이해할 때 도움이 될 거 같아 정리해봤습니다.

temp = torch.LongTensor([0])

temp_em = nn.Embedding(256,4)
print(temp_em(temp))
# tensor([[ 0.7098, -0.5407,  1.8632, -0.5504]], grad_fn=<EmbeddingBackward0>)