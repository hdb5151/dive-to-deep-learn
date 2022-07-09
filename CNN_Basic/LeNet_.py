import torch
from torch import nn
from d2l import torch as d2l

import sys
import os
sys.path.append("_base_py")
import cnn_train as ct  
import epochBASE as eB 

# 计算公式 [(N_h-K_h+P_h+S_h)/S_h] * [(N_w-K_w+P_w+S_w)/S_w]
net = nn.Sequential(
                  nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),    # 6 * 28 * 28
                  nn.AvgPool2d(kernel_size=2, stride=2),    #6 * 14 * 14
                  nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  #16 * 10 * 10
                  nn.AvgPool2d(kernel_size=2, stride=2), #16 * 5 * 5
                  nn.Flatten(),     #将多维向量  变成1维
                  nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                  nn.Linear(120, 84), nn.Sigmoid(),
                  nn.Linear(84, 10))

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
      X = layer(X)
      print(layer.__class__.__name__,'output shape: \t',X.shape)          

batch_size = 256
train_iter, test_iter = eB.load_data_fashion_mnist(batch_size=batch_size)          

lr, num_epochs = 0.9, 10
ct.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()