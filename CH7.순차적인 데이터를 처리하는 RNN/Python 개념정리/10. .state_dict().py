import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.layer1 = nn.Linear(5,2)
        self.layer2 = nn.Linear(2,1)


model = Model()
print(model.state_dict())

# OrderedDict([('layer1.weight', tensor([[ 0.2873,  0.2578,  0.3693, -0.3722,  0.0330],
#         [ 0.3188,  0.4158, -0.2443, -0.3368,  0.2170]])), ('layer1.bias', tensor([-0.2486, -0.2805])), ('layer2.weight', tensor([[ 0.6622, -0.4147]])), ('layer2.bias', tensor([-0.3537]))])