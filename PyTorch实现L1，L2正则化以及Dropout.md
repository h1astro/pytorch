# PyTorch实现L1，L2正则化以及Dropout

## 模型训练中经常出现的两类典型问题：

* 一类是模型无法得到较低的训练误差，我们将这一现象称作**欠拟合**（underfitting）；

* 另一类是模型的训练误差远小于它在测试数据集上的误差，我们称该现象为**过拟合**（overfitting）。

在实践中，我们要尽可能同时应对欠拟合和过拟合。

过拟合现象，即模型的训练误差远小于它在测试集上的误差。虽然增大训练数据集可能会减轻过拟合，但是获取额外的训练数据往往代价高昂。

为此我们可以使用两种方法**权重衰减**（weight decay）、**丢弃法**（dropout）。

## 权重衰减（weight decay）

可以使用L1，L2正则化

正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段。

$L_1$范数惩罚项指的是模型权重参数每个元素的**绝对值**和与一个正的常数的乘积

$L_2$范数惩罚项指的是模型权重参数每个元素的**平方和**与一个正的常数的乘积。举例线性回归损失函数
$$
\ell(w_1, w_2, b) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right)^2
$$
为例，其中$w_1, w_2​$是权重参数，$b​$是偏差参数，样本$i​$的输入为$x_1^{(i)}, x_2^{(i)}​$，标签为$y^{(i)}​$，样本数为$n​$。将权重参数用向量$\boldsymbol{w} = [w_1, w_2]​$表示，

带有$L_1​$范数惩罚项的新损失函数为

$$\ell(w_1, w_2, b) + \frac{\lambda}{n} |\boldsymbol{w}|,$$

带有$L_2​$范数惩罚项的新损失函数为

$$\ell(w_1, w_2, b) + \frac{\lambda}{2n} \|\boldsymbol{w}\|^2,​$$

其中超参数$\lambda > 0​$。当权重参数均为0时，惩罚项最小。当$\lambda​$较大时，惩罚项在损失函数中的比重较大，这通常会使学到的权重参数的元素较接近0。当$\lambda​$设为0时，惩罚项完全不起作用。上式中$L_2​$范数平方$\|\boldsymbol{w}\|^2​$展开后得到$w_1^2 + w_2^2​$。有了$L_2​$范数惩罚项后，在小批量随机梯度下降中，我们将线性回归一节中权重$w_1​$和$w_2​$的迭代方式更改为
$$
\begin{aligned}w_1 &\leftarrow \left(1- \frac{\eta\lambda}{|\mathcal{B}|} \right)w_1 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_1^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\w_2 &\leftarrow \left(1- \frac{\eta\lambda}{|\mathcal{B}|} \right)w_2 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_2^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right).\end{aligned}
$$
**初始化模型参数**

首先，定义随机初始化模型参数的函数。该函数为每个参数都附上梯度。

``` python
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```
**定义$L_1$范数惩罚项**

下面定义$L_1​$范数惩罚项。这里只惩罚模型的权重参数。

``` python
def l1_penalty(w):
    return (torch.abs(w)).sum()
```

**定义$L_2$范数惩罚项**

``` python
def l2_penalty(w):
    return (w**2).sum() / 2
```

### 3.12.3.3 定义训练和测试

下面定义如何在训练数据集和测试数据集上分别训练和测试模型。这里在计算最终的损失函数时添加了$L_2$范数惩罚项。

``` python
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # # 添加了L1范数惩罚项
            # l = loss(net(X, w, b), y) + lambd * l1_penalty(w)
            # 添加了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()
            
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())
       
# 观察过拟合
fit_and_plot(lambd=0)

# 使用权重衰减
fit_and_plot(lambd=3)
```

* 权重衰减可以通过优化器中的`weight_decay`超参数来指定。

## 丢弃法 (Dropout)

![dropout ](https://img-blog.csdnimg.cn/20190816114129116.png)

**丢弃法**简单理解为以一定概率丢弃一些神经元。

设随机变量$\xi_i​$为0和1的概率分别为$p​$和$1-p​$。使用丢弃法时我们计算新的隐藏单元$h_i'​$
$$
h_i' = \frac{\xi_i}{1-p} h_i
$$
由于$E(\xi_i) = 1-p$，因此
$$
E(h_i') = \frac{E(\xi_i)}{1-p}h_i = h_i
$$
由此可看出丢弃发**不改变其输入的期望值**。下面的`dropout`函数将以`drop_prob`的概率丢弃`X`中的元素。

``` python
%matplotlib inline
import torch
import torch.nn as nn
import numpy as np

def dropout(X, drop_prob):
    X = X.float()
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.randn(X.shape) < keep_prob).float()
    
    return mask * X / keep_prob
```

我们运行几个例子来测试一下`dropout`函数。其中丢弃概率分别为0、0.5和1。

``` python
X = torch.arange(16).view(2, 8)
dropout(X, 0)
```

``` python
dropout(X, 0.5)
```

``` python
dropout(X, 1.0)
```

### 简洁实现

在PyTorch中，我们只需要在全连接层后添加`Dropout`层并指定丢弃概率。在训练模型时，`Dropout`层将以指定的丢弃概率随机丢弃上一层的输出元素；在测试模型时（即`model.eval()`后），`Dropout`层并不发挥作用。

```python
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(drop_prob1),
        nn.Linear(num_hiddens1, num_hiddens2), 
        nn.ReLU(),
        nn.Dropout(drop_prob2),
        nn.Linear(num_hiddens2, 10)
        )

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)
```

下面训练并测试模型。

```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```

