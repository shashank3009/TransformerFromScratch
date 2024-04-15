import torch
import torch.nn as nn
# ------------------
import math

def self_attention(x, src_mask):
    return x**2 + src_mask

x = 16
src_mask = 3

sublayer = lambda x: self_attention(x, src_mask)
v = sublayer(math.sqrt(x))
print(v)
# ------------------
x = torch.randn(2, 10)
print(x)
print(x.view(-1))
# ------------------
x = torch.randn(1, 10)
print(x)
y = nn.Dropout(p=0.2)(x)
print(y)
1.7699/0.8
# ------------------
x = torch.rand(1,10,8)
print(x)
x = x.view(x.shape[0], x.shape[1], 2, 4)
print(x)
x=x.transpose(1,2)
print(x)
# ------------------
x = torch.tensor([2,3,4,1,1,1,1])
#print(x.unsqueeze(0).unsqueeze(0).int())
# print(x.unsqueeze(0))
mask = torch.triu(torch.ones(1, 7, 7), diagonal=1).type(torch.int)

print((x!=1).unsqueeze(0).int())
print(mask==0)
print((x!=1).unsqueeze(0).int() & (mask==0))
# ------------------
x = torch.ones(5,4)
print(x.view(-1).size())

import torch.nn.functional as F
predictions = torch.rand(2, 3, 4)
target = torch.randint(9, (2, 3))
print(f"predictions:\n{predictions}\n\n")
print(f"target:\n{target}\n\n")

# Reorder the dimensions
# From: [time_step, batch_size, vocabulary_size]
# To: [batch_size, vocabulary_size, time_step]
predictions = predictions.permute(1, 2, 0)
# From: [time_step, batch_size]
# To: [batch_size, time_step]
target = target.transpose(0, 1)
print(f"predictions size:\n{predictions.size()}\n\n")
print(f"target size:\n{target.size()}\n\n")

print(f"predictions:\n{predictions}\n\n")
print(f"target:\n{target}\n\n")

F.cross_entropy(predictions, target)
# ------------------