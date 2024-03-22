import torch
import torch.optim as optim
import torch.nn.functional as F

from EscapeEnv.common.torch_layers import BaseNetwork

net1 = BaseNetwork(2,4)

net2 = BaseNetwork(2,4).eval()

def param_mse(net1, net2):
    mse = 0
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        mse += F.mse_loss(p1, p2.detach(), reduction="sum")
    
    return mse/2

optimizer = optim.Adam(net2.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = param_mse(net1, net2)
loss.backward()

for p1, p2 in zip(net1.parameters(), net2.parameters()):
    print(p1.grad.data)
    print(p1-p2)
    
for p in net2.parameters():
    print(p.grad.data)