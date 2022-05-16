import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from net import Net
# 创建一个一模一样的模型
model=Net()
# 加载预训练模型的参数
model.load_state_dict(torch.load("Linear.pth"))

# 使用和训练数据一样的数据
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

# 计算loss 的值
inputs=torch.from_numpy(x_train)
target=torch.from_numpy(y_train)
output=model(inputs)
loss_fun=nn.MSELoss()
loss=loss_fun(output,target)
print(loss.item())

# 绘制图形
plt.plot(x_train,y_train,'ro',label='origin data')
plt.plot(x_train,output.detach_().numpy(),label='Fitted data')
plt.legend()
plt.show()
