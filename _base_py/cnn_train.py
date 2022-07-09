import torch
from torch import nn
from d2l import torch as d2l

import sys
import os
sys.path.append("_base_py")
import epochBASE as eB

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
      """使⽤GPU计算模型在数据集上的精度"""
      if isinstance(net, nn.Module):
            net.eval() # 设置为评估模式
      if not device:
            device = next(iter(net.parameters())).device
      # 正确预测的数量，总预测的数量
      metric = eB.Accumulator(2) 
      with torch.no_grad():
            for X, y in data_iter:
                  if isinstance(X, list):
                        # BERT微调所需的（之后将介绍）
                        X = [x.to(device) for x in X]
                  else:
                        X = X.to(device)
                  y = y.to(device)
                  metric.add(eB.accuracy(net(X), y), y.numel())
      return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):#@save
      """⽤GPU训练模型(在第六章定义)"""
      def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                  nn.init.xavier_uniform_(m.weight)
      net.apply(init_weights)
      print('training on', device)
      net.to(device)
      optimizer = torch.optim.SGD(net.parameters(), lr=lr)
      loss = nn.CrossEntropyLoss()
      animator = eB.Animator(xlabel='epoch', xlim=[1, num_epochs],
                              legend=['train loss', 'train acc', 'test acc'])
      timer, num_batches = d2l.Timer(), len(train_iter)
      for epoch in range(num_epochs):
            # 训练损失之和，训练准确率之和，样本数
            metric = eB.Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                  timer.start()
                  optimizer.zero_grad()
                  X, y = X.to(device), y.to(device)
                  y_hat = net(X)
                  l = loss(y_hat, y)
                  l.backward()
                  optimizer.step()
                  with torch.no_grad():
                        metric.add(l * X.shape[0], eB.accuracy(y_hat, y), X.shape[0])
                  timer.stop()
                  train_l = metric[0] / metric[2]
                  train_acc = metric[1] / metric[2]
                  if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                        animator.add(epoch + (i + 1) / num_batches,
                                    (train_l, train_acc, None))
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
      print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}')
      print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(device)}')
