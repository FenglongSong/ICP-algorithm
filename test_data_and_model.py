import ICP
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pointsize = 1

# Reading data points
# data_ = np.load('blade800r.npy')
# model_ = np.load('blade8000.npy')
data_ = np.load('blade8000r.npy')
model_ = np.load('blade.npy')
data = np.mat(data_)
data = np.transpose(data)
model = np.mat(model_[0:3, :])


f1 = plt.figure()
ax = f1.add_subplot(111, projection='3d')
ax.scatter(data[0, :], data[1, :], data[2, :], c='y', linewidths=pointsize, label="data points")  # 绘制数据点
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
# plt.title("data points")


v1 = random.uniform(0, 1)
v2 = random.uniform(0, 1)
v3 = random.uniform(0, 1)
R1 = np.mat([[1, 0, 0], [0, math.cos(v1), -math.sin(v1)], [0, math.sin(v1), math.cos(v1)]])
R2 = np.mat([[math.cos(v2), 0, math.sin(v2)], [0, 1, 0], [-math.sin(v2), 0, math.cos(v2)]])
R3 = np.mat([[math.cos(v3), -math.sin(v3), 0], [math.sin(v3), math.cos(v3), 0], [0, 0, 1]])
R = R3 * R2 * R1

model = 1.1 * R * model
model[0, :] += v1
model[1, :] += v3
model[2, :] += v2

ax.scatter(model[0, :], model[1, :], model[2, :], c='b', linewidths=pointsize, label="model points")  # 绘制数据
# plt.title("transformed points")

# print(data.shape, model.shape)
# # Begin to find the matching points

# init = np.zeros([3, 3])
init = np.eye(3)
data_transformed = ICP.iterate(data, model, init, 200, 0.001)
# 如果输入的npy文件的单位是mm，那么0.001就是1μm

# ax1 = f1.add_subplot(111, projection='3d')
ax.scatter(data_transformed[0, :], data_transformed[1, :], data_transformed[2, :], c='r', linewidths=pointsize, label="transformed data points")  # 绘制数据
# plt.title("transformed points")
plt.show()

