#导入必要的模块
import numpy as np
import matplotlib.pyplot as plt

# 一.1、python画图之散点图sactter函数基本的使用方法
def scatter_demo1():
    """

    :return:
    """
    # 产生测试数据
    x = np.arange(1, 10)
    y = x
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Scatter Plot')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    ax1.scatter(x, y, c='r', marker='o')
    # 设置图标
    plt.legend('x1')
    # 显示所画的图
    plt.show()


# 一.2、python画图之散点图sactter函数不同大小
def scatter_demo2():
    """

    :return:
    """
    # 产生测试数据
    x = np.arange(1, 10)
    y = x
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Scatter Plot')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    sValue = x * 10
    ax1.scatter(x, y, s=sValue, c='r', marker='x')
    # 设置图标
    plt.legend('x1')
    # 显示所画的图
    plt.show()


# 一.3、python画图之散点图sactter函数不同颜色
def scatter_demo3():
    """

    :return:
    """
    # 产生测试数据
    x = np.arange(1, 10)
    y = x
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Scatter Plot')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    cValue = ['r', 'y', 'g', 'b', 'r', 'y', 'g', 'b', 'r']
    ax1.scatter(x, y, c=cValue, marker='s')
    # 设置图标
    plt.legend('x1')
    # 显示所画的图
    plt.show()

# 一.4、python画图之散点图sactter函数 线宽linewidths
def scatter_demo4():
    """

    :return:
    """
    # 产生测试数据
    x = np.arange(1, 10)
    y = x
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Scatter Plot')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    lValue = x
    ax1.scatter(x, y, c='r', s=100, linewidths=lValue, marker='o')
    # 设置图标
    plt.legend('x1')
    # 显示所画的图
    plt.show()


# 二、numpy 中 newaxis函数的使用
def newaxis_demo():
    """
    newaxis表示增加一个新的坐标轴，不好理解，看例子就明白了
    :return:
    """
    # import numpy as np
    a = np.array([1, 2, 3])
    print("a:\n", a.shape)    # (3,)
    print(a)          # [1 2 3]

    b = np.array([1, 2, 3])[:, np.newaxis]
    print("b:\n", b.shape, '\n', b)   # (3, 1)

    c = np.array([1, 2, 3])[np.newaxis, :]
    print("c:\n", c.shape, '\n', c)   # (1, 3)



if __name__ == '__main__':
    # 代码一.1：python画图之散点图sactter函数基本的使用方法
    # scatter_demo1()

    # 代码一.2：python画图之散点图sactter函数不同大小
    # scatter_demo2()

    # 代码一.3：python画图之散点图sactter函数不同颜色
    # scatter_demo3()

    # 代码一.4：python画图之散点图sactter函数线宽linewidths
    # scatter_demo4()

    # 代码二：numpy 中 newaxis函数的使用
    newaxis_demo()