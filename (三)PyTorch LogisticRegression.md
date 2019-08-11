## 1.PyTorch基础实现代码

```python
import torch
from torch.autograd import Variable
 
torch.manual_seed(2)
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.0], [0.0], [1.0], [1.0]]))
 
# 初始化
w = Variable(torch.Tensor([-1]), requires_grad=True)
b = Variable(torch.Tensor([0]), requires_grad=True)
 
epochs = 100
costs = []
lr = 0.1
 
print('before training,predict of x = 1.5 is :')
print('Y_pred = ', float(w.data * 1.5 + b.data > 0))

 
# 模型训练
for epoch in range(epochs):
    # 计算梯度
    A = 1 / (1 + torch.exp(-(w * x_data + b)))
    # 逻辑损失函数
    J = - torch.mean(y_data * torch.log(A) + (1 - y_data) * torch.log(1 - A))
    # 自动反向传播
    J.backward()
 
    # 参数更新
    w.data = w.data - lr * w.grad.data
    w.grad.data.zero_()
    b.data = b.data - lr * b.grad.data
    b.grad.data.zero_()
 
# 模型测试
print('after trainning,predict of x = 1.5 is :')
print('Y_pred = ', float(w.data * 1.5 + b.data > 0))
print(w.data, b.data)

```



## 2. 用PyTorch类实现Logistic regression,torch.nn.module写网络结构

```python
import torch
# from torch import nn
# 第一创建数据
from torch.autograd import Variable  # 导入Variable函数进行自动求导，有了Variable PyTorch才能实现自动求导功能
 
torch.manual_seed(2)
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.0], [0.0], [1.0], [1.0]]))
 
 
# 定义网络模型
# 先建立一个基类Model，都是从父类torch.nn.Module中继承过来，PyTorch写网络的固定写法
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # 初始父类
        self.linear = torch.nn.Linear(1, 1)  # 输入维度和输出维度都为1
 
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
 
 
model = Model()  # 实例化
 
# 定义Loss和优化方法
criterion = torch.nn.BCEWithLogitsLoss()  # 损失函数，封装好的逻辑损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 进行优化梯度下降
# before training
hour_var = Variable(torch.Tensor([[2.5]]))
y_pred = model(hour_var)
print("predict (before training) given", 4, "is", float(model(hour_var).data[0][0] > 0.5))
 
epochs = 40
for epoch in range(epochs):
    # 计算grads and cost
    y_pred = model(x_data)  # x_data 输入数据进入模型中
    loss = criterion(y_pred, y_data)
    # print(loss.data)
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 优化迭代
 
# after trining
hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = model(hour_var)  # 预测结果
print("predict (after training) given", 4, "is", float(model(hour_var).data[0][0] > 0.5))

```

