import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from IPython import display
from d2l import torch as d2l

# 加载/下载 数据集

# Defined in file: ./chapter_linear-networks/image-classification-dataset.md
def get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

# 计算分类的精度
def accuracy(y_hat,y):#@save
      """计算预测正确的数量"""
      if len(y_hat.shape)>1 and y_hat.shape[1]>1:
            # print("----y_hat.shape=====",y_hat)
            y_hat=y_hat.argmax(axis=1)    #   原始  y_hat.shape=[256,10]             y_hat.argmax(axis=1)  返回每一行最大值的下标. 最后y_hat.shape=[256]
            # print("----y_hat=====",y_hat)
      cmp=y_hat.type(y.dtype)==y    #     首先转换y_hat的类型使之与y.dtype一样(torch.int64),然后逐一比较y_hat和y向量的值,如果对应的值相同.则为true(1),否则为false(0)
      # print(y_hat.type(y.dtype))  #     y.dtype=torch.int64
      return float(cmp.type(y.dtype).sum())     #     返回每批次中y_hat中有多少项与y相同

def evaluate_accuracy(net,data_iter):#@save
      """计算在指定数据集上模型的精度"""
      if isinstance(net,torch.nn.Module):
            net.eval()  # 将模型设置为评估模式
      metric=Accumulator(2)   # 正确预测数、预测总数
      with torch.no_grad():
            for X,y in data_iter:
                  metric.add(accuracy(net(X),y),y.numel())  #     y.numel()==256  表示张量里含有多少元素
                  # print(accuracy(net(X),y),'---',y.numel())
                  # print(metric[0],'===',metric[1])
      return metric[0]/metric[1]

class Accumulator: #@save
      """在n个变量上累加"""
      def __init__(self, n):
            self.data = [0.0] * n   # 返回 矩阵 矩阵大小由 n 决定
      def add(self, *args):
            self.data = [a + float(b) for a, b in zip(self.data, args)] # 对输入的矩阵 arg 进行迭代累加
      def reset(self):
            self.data = [0.0] * len(self.data)
      def __getitem__(self, idx):
            return self.data[idx]

# 训练
def train_epoch_ch3(net,train_iter,loss,updater):#@save
      """训练模型⼀个迭代周期（定义⻅第3章）"""
      # 将模型设置为训练模式
      if isinstance(net,torch.nn.Module): #isinstance(object,classinfo):函数来判断一个对象是否是一个已知的类型...如果 net 是torch.nn.Module类型
            net.train()                   #保证BN层能够用到每一批数据的均值和方差# 
      # 训练损失总和,训练准备准确总和,样本数
      metric=Accumulator(3)
      for X,y in train_iter:   #    X.shape=[256,1,28,28] --- y.shape=[256,1]     
      # 计算梯度更新参数
            y_hat=net(X)      #     y_hat.shape=[256,10]
            # print(y_hat.shape)
            l=loss(y_hat,y)   # 计算交叉熵      l.shape=[256]
            if isinstance(updater,torch.optim.Optimizer):
                  # 使用PyTorch 内置的优化器和损失函数
                  updater.zero_grad()
                  l.mean().backward()
                  updater.step()
            else:
                  # 使用定制的优化器的损失函数
                  l.sum().backward()
                  updater(X.shape[0])
            metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
      # 返回训练损失和训练精度
      return metric[0]/metric[2],metric[1]/metric[2]


class Animator: #@save
      """在动画中绘制数据"""
      def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,ylim=None,
                  xscale='linear', yscale='linear',
                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, 
                  ncols=1,figsize=(3.5, 2.5)):
            # 增量地绘制多条线
            if legend is None:
                  lengend=[]
            d2l.use_svg_display()
            self.fig,self.axes=d2l.plt.subplots(nrows,ncols,figsize=figsize)
            if nrows*ncols==1:
                  self.axes=[self.axes]
            #使用lambda函数捕获参数
            self.config_axes=lambda :d2l.set_axes(
                  self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend)
            self.X,self.Y,self.fmts=None,None,fmts
      
      def add(self,x,y):
            # 向图表中添加多个数据点
            if not hasattr(y,"__len__"):
                  y=[y]
            n=len(y)
            if not hasattr(x,"__len__"):
                  x=[x]*n
            if not self.X:
                  self.X = [[] for _ in range(n)]
            if not self.Y:
                  self.Y = [[] for _ in range(n)]
            for i, (a, b) in enumerate(zip(x, y)):
                  if a is not None and b is not None:
                        self.X[i].append(a)
                        self.Y[i].append(b)
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                  self.axes[0].plot(x, y, fmt)
            self.config_axes()

            display.display(self.fig)
            display.clear_output(wait=True)
            # d2l.plt.show()

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
      """训练模型（定义⻅第3章）"""
      animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
      legend=['train loss', 'train acc', 'test acc'])
      for epoch in range(num_epochs):
            train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
      train_loss, train_acc = train_metrics
      assert train_loss < 0.5, train_loss
      assert train_acc <= 1 and train_acc > 0.7, train_acc
      assert test_acc <= 1 and test_acc > 0.7, test_acc