
# Pytorch

## 一、什么是Pytorch，为什么选择Pytroch？

PyTorch 是一个基于 Python 的科学计算包，主要定位两类人群：

- NumPy 的替代品，可以利用 GPU 的性能进行计算。
- 深度学习研究平台拥有足够的灵活性和速度
- 动态神经网络
- Python优先
- 命令式体验
- 轻松扩展

## 二、Pytroch的安装(CPU版本)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190807184458112.png)

登录官网https://pytorch.org/

输入自己需要安装的环境，我是win10操作系统，Conda的管理环境，Python3.7，没有安装cuda，选择无，即cpu版本。 gpu版本等以后再改，新手可以先入门cpu版本的。

```
conda install pytorch-cpu torchvision-cpu -c pytorch
```

- 如何判断安装成功？

  在python里输入`import torch`，如果没有报错则说明安装成功

## 七、通用代码实现流程(实现一个深度学习的代码流程)

PyTorch搭建神经网络的一般步骤

- 给定输入输出
- 定义一个模型
- 定义损失函数（loss function）和优化函数（optimizer）
- 训练一个过程

#### 手写数字识别  代码源自[开源书籍](https://github.com/zergtant/pytorch-handbook)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
torch.__version__
```

首先，我们定义一些超参数

```python
BATCH_SIZE=512 #大概需要2G的显存
EPOCHS=20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
```

因为Pytorch里面包含了MNIST的数据集，所以我们这里直接使用即可。 如果第一次执行会生成data文件夹，并且需要一些时间下载，如果以前下载过就不会再次下载了

由于官方已经实现了dataset，所以这里可以直接使用DataLoader来对数据进行读取

```python
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
```

测试集

```python
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
```

下面我们定义一个网络，网络包含两个卷积层，conv1和conv2，然后紧接着两个线性层作为输出，最后输出10个维度，这10个维度我们作为0-9的标识来确定识别出的是那个数字

在这里建议大家将每一层的输入和输出维度都作为注释标注出来，这样后面阅读代码的会方便很多

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1=nn.Conv2d(1,10,5) # 10, 24x24
        self.conv2=nn.Conv2d(10,20,3) # 128, 10x10
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x) #24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  #12
        out = self.conv2(out) #10
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out
```

我们实例化一个网络，实例化后使用.to方法将网络移动到GPU

优化器我们也直接选择简单暴力的Adam

```python
model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())
```

下面定义一下训练的函数，我们将训练的所有操作都封装到这个函数中

```python
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
```

测试的操作也一样封装成一个函数

```python
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

下面开始训练，这里就体现出封装起来的好处了，只要写两行就可以了

```python
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
```

我们看一下结果，准确率99%，没问题

如果你的模型连MNIST都搞不定，那么你的模型没有任何的价值

即使你的模型搞定了MNIST，你的模型也可能没有任何的价值

MNIST是一个很简单的数据集，由于它的局限性只能作为研究用途，对实际应用带来的价值非常有限。但是通过这个例子，我们可以完全了解一个实际项目的工作流程

我们找到数据集，对数据做预处理，定义我们的模型，调整超参数，测试训练，再通过训练结果对超参数进行调整或者对模型进行调整。

并且通过这个实战我们已经有了一个很好的模板，以后的项目都可以以这个模板为样例
