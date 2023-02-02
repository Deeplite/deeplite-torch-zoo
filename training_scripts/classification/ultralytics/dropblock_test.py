import torch
from dropblock import DropBlock2D

x = torch.rand(100,100,16,16)
drop_block = DropBlock2D(block_size=3, drop_prob=0.3)
regularized_x = drop_block(x)

print (regularized_x)

