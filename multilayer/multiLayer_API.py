# 多层感知机 API实现
from matplotlib.pyplot import show
import torch
from torch import nn
from d2l import torch as d2l

import sys
import os
sys.path.append("../_base_py")
import epochBASE as eB


batch_size, lr, num_epochs = 256, 0.1, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net=nn.Sequential(nn.Flatten(),     #     nn.Flatten(x) 将张量拉成1维, 即数据扁平化
                  nn.Linear(784,256),
                  nn.ReLU(),
                  nn.Linear(256,10))

def init_weight(m):
      if type(m)==nn.Linear:
            nn.init.normal_(m.weight,std=0.01)  # 没看懂

net.apply(init_weight);                        

loss = nn.CrossEntropyLoss(reduction='none')         #计算 损失

trainer = torch.optim.SGD(net.parameters(), lr=lr)

# if __name__ =='__main__':
if __name__== '__main__':
      eB.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
      d2l.plt.show()

