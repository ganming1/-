import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 一、载入数据
def load_dates():
    data = np.genfromtxt("job.csv", delimiter=",")
    x_data = data[1:, 1]
    y_data = data[1:, 2]

    # 画散点图显示
    plt.scatter(x_data, y_data)
    plt.show()
    # 数据加维
    x_data = x_data[:, np.newaxis]
    y_data = y_data[:, np.newaxis]

    return x_data, y_data


# 二、创建线性拟合模型
def linear_regression():
    x_data, y_data =load_dates()
    # 创建拟合模型
    model = LinearRegression()
    model.fit(x_data, y_data)    # 要二维数据传入，所以上面要加维
    # 画图
    plt.plot(x_data, y_data, 'b.')
    plt.plot(x_data, model.predict(x_data), 'r')
    plt.show()

    return None


# 三、定义多项式回归,degree的值可以调节多项式的特征
def polynomial_features():
    x_data, y_data = load_dates()
    print(x_data)
    poly_reg = PolynomialFeatures(degree=5)
    # 使特征多degree列
    x_poly = poly_reg.fit_transform(x_data)
    print(x_poly)
    # 定义回归模型
    lin_reg = LinearRegression()
    # 训练模型
    lin_reg.fit(x_poly, y_data)

    # 画图 不够平滑是因为用训练数据来画的线，训练数据比较少
    plt.plot(x_data, y_data, 'b.')
    plt.plot(x_data, lin_reg.predict(poly_reg.fit_transform(x_data)), c='r')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

    # 画图 用自己生成的100个测试数据画线，会比较平滑
    plt.plot(x_data, y_data, 'b.')
    x_test = np.linspace(1, 10, 500)
    x_test = x_test[:, np.newaxis]
    plt.plot(x_test, lin_reg.predict(poly_reg.fit_transform(x_test)), c='r')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

if __name__ == '__main__':
    # 代码一、载入数据
    # load_dates()

    # 代码二、创建线性拟合模型
    # linear_regression()

    # 代码三、定义多项式回归,degree的值可以调节多项式的特征
    polynomial_features()
