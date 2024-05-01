### RNN ¸ðµ¨ ±¸Çö ###

### BasicGRU¶ó´Â RNN ¸ðµ¨ ±¸Çö ###

import torch
import torch.nn as nn

class BasicGRU(nn.Module):
    def __init__(self,n_layers,hidden_dim,n_vocab,embed_dim,n_classes,dropout_p=0.2):
        super(BasicGRU,self).__init__()
        print("Building Basic GRU model...")