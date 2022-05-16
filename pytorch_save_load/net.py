import torch
import torch.nn as nn

"""
net.py  用于定义网络的结构
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1=nn.Linear(1,3)
        self.layer2=nn.Linear(3,1)
    def forward(self,x):
        x=self.layer1(x)
        x=torch.relu(x)
        x=self.layer2(x)
        return x
