import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import PolynomialFeatures

# 一、生成数据，显示散点图
def get_data_show():
    """
    生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征
    可以生成两类或多类数据
    :return:
    """
    x_data, y_data = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
    # plt.show()
    return x_data, y_data


# 二、线性回归模型，及作图显示
def xianxin_show():
    x_data, y_data = get_data_show()

    logistic = linear_model.LogisticRegression()
    logistic.fit(x_data, y_data)

    # 获取数据值所在的范围
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    # 生成网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    # ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
    z = logistic.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # 等高线图
    cs = plt.contourf(xx, yy, z)
    # 样本散点图
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
    plt.show()
    print('score:', logistic.score(x_data, y_data))  # score: 0.514


# 三、多项式回归,及作图显示
def duoxiangshi_show():
    x_data, y_data = get_data_show()
    poly_reg = PolynomialFeatures(degree=5)
    # 特征处理
    x_poly = poly_reg.fit_transform(x_data)
    # 定义逻辑回归模型
    logistic = linear_model.LogisticRegression()
    # 训练模型
    logistic.fit(x_poly, y_data)

    # 获取数据值所在的范围
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    # 生成网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    z = logistic.predict(poly_reg.fit_transform(np.c_[xx.ravel(), yy.ravel()]))# ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
    z = z.reshape(xx.shape)
    # 等高线图
    cs = plt.contourf(xx, yy, z)
    # 样本散点图
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
    plt.show()

    print('score:', logistic.score(x_poly, y_data))  # score: 0.982

if __name__ == '__main__':
    # 代码1：生成数据，显示散点图
    # get_data_show()

    # 代码2：线性回归模型，及作图显示
    # xianxin_show()

    # 代码3：多项式回归,及作图显示
    duoxiangshi_show()

