# 5.1.0
import torch
from torch import nn
from torch.nn import functional as F
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
# print(X.shape)
#                                          -----------------------------------
#5.1.1 自定义块
class MLP(nn.Module):
      # ⽤模型参数声明层。这⾥，我们声明两个全连接的层
      def __init__(self):
            # 调⽤MLP的⽗类Module的构造函数来执⾏必要的初始化。
            # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
            super().__init__()
            self.hidden = nn.Linear(20, 256) # 隐藏层
            self.out = nn.Linear(256, 10) # 输出层
      # 定义模型的前向传播，即如何根据输⼊X返回所需的模型输出
      def forward(self, X):
            # 注意，这⾥我们使⽤ReLU的函数版本，其在nn.functional模块中定义。
            return self.out(F.relu(self.hidden(X)))

net=MLP()   # 自动调用init()初始化
print("5.1.1 自定义块-->",net(X),"\n",
      "net(X).shape == ",net(X).shape)       

#                                          -----------------------------------

# 5.1.2 顺序块

class MySequential(nn.Module):
      def __init__(self, *args):
            super().__init__()
            for idx, module in enumerate(args):       #enumerate: 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                  # 这⾥，module是Module⼦类的⼀个实例。我们把它保存在'Module'类的成员
                  # 变量_modules中。module的类型是OrderedDict
                  self._modules[str(idx)] = module
      def forward(self, X):
            # OrderedDict保证了按照成员添加的顺序遍历它们
            for block in self._modules.values():
                  X = block(X)
            return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print("5.1.2 顺序块-->",net(X),"\n",
      "net(X).shape == ",net(X).shape)   

#                                          -----------------------------------

# 5.1.3 在前向传播函数中执⾏代码

class FixedHiddenMLP(nn.Module):
      def __init__(self):
            super().__init__()
            # 不计算梯度的随机权重参数。因此其在训练期间保持不变
            self.rand_weight = torch.rand((20, 20), requires_grad=False)
            self.linear = nn.Linear(20, 20)
      def forward(self, X):
            X = self.linear(X)
            # 使⽤创建的常量参数以及relu和mm函数
            X = F.relu(torch.mm(X, self.rand_weight) + 1)   #torch.mm(a,b) :  矩阵 a,b 相乘 
            # 复⽤全连接层。这相当于两个全连接层共享参数
            X = self.linear(X)
            # 控制流
            while X.abs().sum() > 1:
                  X /= 2
            return X.sum()




