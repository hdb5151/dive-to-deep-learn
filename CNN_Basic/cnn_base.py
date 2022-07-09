import torch
from torch import nn
from d2l import torch as d2l

import sys
import os
sys.path.append("../_base_py")
import epochBASE as eB  

def corr2d(X, K): #@save
      """计算⼆维互相关运算"""
      h, w = K.shape
      Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))   # 经 卷积核后的尺寸
      for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                  Y[i, j] = (X[i:i + h, j:j + w] * K).sum() # 卷积后...
      return Y

# 定义卷积层
class Conv2D(nn.Module):
      def __init__(self, kernel_size):
            super().__init__()
            self.weight = nn.Parameter(torch.rand(kernel_size))
            self.bias = nn.Parameter(torch.zeros(1))
      def forward(self, x):
            return corr2d(x, self.weight) + self.bias

X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
print(K.shape)