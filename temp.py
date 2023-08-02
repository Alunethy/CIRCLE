import numpy as np
import matplotlib.pyplot as plt

# 定义x,y
X = np.linspace(0, 2 * np.pi, 32, endpoint=True)
C = np.cos(X)

# figure的名称
plt.figure('demon plot')
# 画图
plt.plot(X, C, 'r--', label=['cos', "12", "djf", "id"])

# 显示x、y坐标代表的含义
plt.xlabel('Independent variable x')
plt.ylabel('dependent variable y')

# 显示图的标题
plt.title(' demo')

# 显示图例

num1 = 1.05
num2 = 0
num3 = 3
num4 = 0
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

plt.show()