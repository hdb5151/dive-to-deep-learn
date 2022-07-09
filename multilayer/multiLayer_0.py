"""多层感知机---从0实现"""
from matplotlib.pyplot import show
import torch
from torch import nn
from d2l import torch as d2l

import sys
import os
sys.path.append("../_base_py")
import epochBASE as eB

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hidden=784,10,256
# W1.shape=[784,256]
W1=nn.Parameter(torch.randn(num_inputs,num_hidden,requires_grad=True)*0.01)         # 返回符合均值为0,方差为1 正态分布的 num_inputs * num_hiddn 的矩阵
#b1.shape=[256]
b1=nn.Parameter(torch.zeros(num_hidden,requires_grad=True))
# W2.shape=[256,10]
W2=nn.Parameter(torch.randn(num_hidden,num_outputs,requires_grad=True)*0.01)
#b1.shape=[10]
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params = [W1, b1, W2, b2]

# 激活函数
def relu(X):
      a=torch.zeros_like(X)   #a.shape=[]
      return torch.max(X,a)


# 网络模型
def net(X): #X.shape[256,1,28,28]
      X=X.reshape((-1,num_inputs))        # 最终 X.shape=[256,784]
      H=relu(X@W1+b1)   # 这⾥“@”代表矩阵乘法        #H.shape=[256,256]
      return (H@W2+b2)  #返回数据shape=[256,10]  256==batch-size  10==类别


#损失函数
loss = nn.CrossEntropyLoss(reduction='none')    #交叉熵

num_epochs,lr=10,0.1
updater=torch.optim.SGD(params,lr=lr)

if __name__ =='__main__':
      eB.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
      d2l.plt.show()
