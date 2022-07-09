print("\n-------------5.5读写文件-------------")

import torch
from torch import nn
from torch.nn import functional as F

print("\n-------------加载和保存张量-------------")
x = torch.arange(4)
torch.save(x, '5-MLP/x-file')
x2 = torch.load('5-MLP/x-file')

print("\n-------------存储⼀个张量列表，然后把它们读回内存-------------")
y = torch.zeros(4)
torch.save([x, y],'5-MLP/x-files')
x2, y2 = torch.load('5-MLP/x-files')
print(x2, y2)

print("\n-------------以写⼊或读取从字符串映射到张量的字典-------------")
mydict = {'x': x, 'y': y}
torch.save(mydict, '5-MLP/mydict')
mydict2 = torch.load('5-MLP/mydict')
print(mydict2)

print("\n------------- 5.5.2 加载和保存模型参数 -------------")
class MLP(nn.Module):
      def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(20, 256)
            self.output = nn.Linear(256, 10)
      def forward(self, x):
            return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 保存模型
torch.save(net.state_dict(),'5-MLP/mlp.params')

# 恢复模型
clone = MLP()
clone.load_state_dict(torch.load('5-MLP/mlp.params'))
print(clone.eval())


print("\n------------- -------------")


print("\n------------- -------------")


print("\n------------- -------------")


print("\n------------- -------------")


print("\n------------- -------------")


print("\n------------- -------------")


print("\n------------- -------------")


print("\n------------- -------------")


print("\n------------- -------------")


print("\n------------- -------------")

