# 5.2 参数管理

import torch
from torch import nn
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(4, 4))

print("net(X)-->",net(X))
# 5.2.1 参数访问
print("\n-------------另一种访问网络参数的方式-------------")
print("net.state_dict()-->",net.state_dict())   # net.state_dict()  返回每一层的状态字典

# 提取目标参数
print("\n-------------另一种访问网络参数的方式-------------")
print("type(net[2].bias)-->",type(net[2].bias))
print("net[2].bias-->",net[2].bias)
print("net[2].bias.data-->",net[2].bias.data)

# ⼀次性访问所有参数
print("\n-------------另一种访问网络参数的方式-------------")
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print("\n-------------另一种访问网络参数的方式-------------")
print(net.state_dict()['2.bias'].data)


print("\n-------------从嵌套块收集参数-------------")
def block1():
      return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                              nn.Linear(8, 4), nn.ReLU())
def block2():
      net = nn.Sequential()
      for i in range(4):
            # 在这⾥嵌套
            net.add_module(f'block {i}', block1())
      return net
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print("rgnet(X)-->",rgnet(X))
print("rgnet-->",rgnet)

print("\n-------------访问第⼀个主要的块中、第⼆个⼦块的第⼀层的偏置项-------------")
print(rgnet[0][1][0].bias.data)

print("\n-------------5.2.2 参数初始化-------------")
      # 内置初始化
def init_normal(m):
      if type(m) == nn.Linear:
            # nn.init.normal_(m.weight, mean=0, std=0.01)         # 权重初始化 标准   均值为 0   方差为 0.01 
            nn.init.constant_(m.weight, 1)                        # 权重初始化 全 '1'
            nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])

def xavier(m):
      if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
def init_42(m):
      if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 42)
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

print("\n-------------⾃定义初始化-------------")
def my_init(m):
      if type(m) == nn.Linear:
            print("Init", *[(name, param.shape)
                              for name, param in m.named_parameters()][0])    
            nn.init.uniform_(m.weight, -10, 10)
            m.weight.data *= m.weight.data.abs() >= 5
net.apply(my_init)
print(net[0].weight[:2])

print("\n-------------参数设置-------------")
net[0].weight.data[:] += 1    # 每行的weight + 1
net[0].weight.data[0, 0] = 42 # 第一行  第一列 weight + 1
print(net[0].weight.data[0])

print("\n-------------参数绑定-------------")
# 我们需要给共享层⼀个名称，以便可以引⽤它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                  shared, nn.ReLU(),
                  shared, nn.ReLU(),
                  nn.Linear(8, 1))
net(X)

print("\n-------------检查参数是否相同-------------")
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同⼀个对象，⽽不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])