import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


# 一、载入数据
def load_dates():
    data = np.genfromtxt("data.csv", delimiter=",")
    x_data = data[:, 0, np.newaxis]
    y_data = data[:, 1, np.newaxis]
    # 画散点图
    # plt.scatter(x_data, y_data)
    plt.plot(x_data, y_data, 'b.')
    # plt.show()
    # mat:将数组转换成矩阵的形式
    print(np.mat(x_data).shape)
    print(np.mat(y_data).shape)
    # 给样本添加偏置项，即加一列 1
    X_data = np.concatenate((np.ones((100, 1)), x_data), axis=1)
    print(X_data.shape)
    print(X_data[:5])

    return X_data, y_data

# 二、标准方程法求解回归参数
def weights(xArr, yArr):
    """
    根据 06图1 求解
    :param xArr: X 矩阵
    :param yArr: y 矩阵
    :return: 参数矩阵 ws
    """
    # mat:将数组转换成矩阵的形式
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat     # 矩阵乘法
    # 计算矩阵的值,如果值为0，说明该矩阵没有逆矩阵
    if np.linalg.det(xTx) == 0.0:
        print("该矩阵不能求逆矩阵")
        return
    # xTx.I 为 xTx 的逆矩阵
    ws = xTx.I * xMat.T * yMat
    return ws

# 三、画图
def apply_demo():
    # 一、载入数据
    x_data, y_data = load_dates()
    # 二、标准方程法求解回归参数
    ws = weights(x_data, y_data)
    print(ws)
    # print(x_data)
    # 两个测试点，用于画直线
    x_test = np.array([[20], [80]])
    y_test = ws[0] + x_test * ws[1]
    # 下一句出现错误图示，因为 x_data 是二维数组，传入plot会转换成一维
    # plt.plot(x_data, y_data, 'b.')
    plt.plot(x_test, y_test, 'r')
    # plt.xlim(20, 100)
    plt.show()


if __name__ == '__main__':
    apply_demo()
