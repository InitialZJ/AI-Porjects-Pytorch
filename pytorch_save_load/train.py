import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from net import Net
# 训练两千次
num_epoches=2000
# 学习率定义为0.01
learning_rate=0.01
# 创造一堆散点（用于训练神经网络）
x_train=np.array([
    3.3,4.4,5.5,6.71,6.93,4.168,
    9.779,6.182,7.59,2.167,7.042,
    10.791,5.313,7.997,3.1
],dtype=np.float32).reshape(-1,1)
y_train=np.array([
    1.7,2.76,2.09,3.19,1.694,1.573,
    3.366,2.596,2.53,1.221,2.827,
    3.465,1.65,2.904,1.3
],dtype=np.float32).reshape(-1,1)
# 创建一个模型
model=Net()
# 使用平方差均值来作为损失函数
loss_fun=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

# 开始训练：
inputs=torch.from_numpy(x_train)
targets=torch.from_numpy(y_train)
for epoch in range(num_epoches):
    output=model(inputs)
    loss=loss_fun(output,targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10==0:
        print("Epoch {} / {},loss {:.4f}".format(epoch+1,num_epoches,loss.item()))

# 保存训练的模型：
torch.save(model.state_dict(),"Linear.pth")

# 打印最终的损失值
output=model(inputs)
loss=loss_fun(output,targets)
print(loss.item())

# 绘制经过网络之后的预测图形和原本的图形，使用matplot 绘制曲线
predicted=model(torch.from_numpy(x_train)).detach_().numpy()
plt.plot(x_train,y_train,'ro',label="Origin data")
plt.plot(x_train,predicted,label="Fitted data")
plt.legend()
plt.show()
