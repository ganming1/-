import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

# 读入数据
data = genfromtxt(r"longley.csv", delimiter=',')
print("data\n", data)
# 切分数据
x_data = data[1:, 2:]
y_data = data[1:, 1, np.newaxis]
print("x_data\n", x_data)
print("y_data\n", y_data)
print("x_data矩阵形式形状\n", np.mat(x_data).shape)
print("y_data矩阵形式形状\n", np.mat(y_data).shape)
# 给样本添加偏置项
X_data = np.concatenate((np.ones((16, 1)), x_data), axis=1)
print(X_data.shape)


# 岭回归标准方程法求解回归参数
def weights(xArr, yArr, lam=0.2):
    """
    岭回归标准方程法求解回归参数
    :param xArr: X 矩阵
    :param yArr: y 矩阵
    :param lam: 岭系数
    :return:
    """
    xMat = np.mat(xArr)    # 转换成矩阵形式
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat     # 矩阵乘法 得到（7，7）

    # np.eye(xMat.shape[1]) 生成7行7列的单位矩阵，xMat.shape[1] = 7.
    rxTx = xTx + np.eye(xMat.shape[1])*lam
    # 计算矩阵的值,如果值为0，说明该矩阵没有逆矩阵
    if np.linalg.det(rxTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    # xTx.I  为xTx的逆矩阵
    ws = rxTx.I * xMat.T * yMat
    return ws

ws = weights(X_data,y_data)
print("求解的系数为：\n", ws)

# 计算预测值
result = np.mat(X_data) * np.mat(ws)
print("X_data矩阵形式：\n", np.mat(X_data))
print("预测结果为：\n", result)
print("真实结果为：\n", y_data)