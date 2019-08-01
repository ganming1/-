import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report

# 数据是否需要标准化
scale = False


# 一、载入数据
def get_data():
    data = np.genfromtxt("LR-testSet2.txt", delimiter=",")
    """print(data[:5, :], type(data))
    [[ 0.051267  0.69956   1.      ]
     [-0.092742  0.68494   1.      ]
     [-0.21371   0.69225   1.      ]
     [-0.375     0.50219   1.      ]
     [-0.51325   0.46564   1.      ]]<class 'numpy.ndarray'>
    """
    x_data = data[:, :-1]
    """print(x_data[:5, :], type(data))
    [[ 0.051267  0.69956 ]
     [-0.092742  0.68494 ]
     [-0.21371   0.69225 ]
     [-0.375     0.50219 ]
     [-0.51325   0.46564 ]]<class 'numpy.ndarray'>
    """
    y_data = data[:, -1, np.newaxis]
    """print(y_data[:5], type(data))
    [[1.]
     [1.]
     [1.]
     [1.]
     [1.]]<class 'numpy.ndarray'>
    """
    return x_data, y_data


# 二、散点图显示数据
def date_show():
    x_data, y_data = get_data()
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    # 切分不同类别的数据
    """print(y_data[0], type(y_data[0]))
    [1.] <class 'numpy.ndarray'>
    """
    for i in range(len(x_data)):
        if y_data[i] == 0:
            """
            此处是 numpy.ndarray 与 float 进行值比较，可以比较
            注意与01_文件中做比较
            """
            x0.append(x_data[i, 0])
            y0.append(x_data[i, 1])
        else:
            x1.append(x_data[i, 0])
            y1.append(x_data[i, 1])

    # 画图
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    # 画图例
    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')
    # plt.show()


# 三、特征处理, 使特征添加多项非线性特征
def poly_demo():
    x_data, y_data = get_data()

    # 定义多项式回归,degree的值可以调节多项式的特征
    poly_reg = PolynomialFeatures(degree=3)

    # 特征处理, 使特征添加多项非线性特征
    x_poly = poly_reg.fit_transform(x_data)

    # 演示PolynomialFeatures()的功能
    def show_poly():
        """
        演示PolynomialFeatures()的功能
        """
        test = [[2, 3]]
        # 定义多项式回归,degree的值可以调节多项式的特征
        poly = PolynomialFeatures(degree=2)
        # 特征处理
        x_poly = poly.fit_transform(test)
        """degree=3时print(x_poly)
        [[ 1.  2.  3.  4.  6.  9.  8. 12. 18. 27.]]
        degree=2时print(x_poly)
        [[1. 2. 3. 4. 6. 9.]]
        """
        print(x_poly)
        return None
    # show_poly()

    return x_poly, y_data, poly_reg


# 四、定义 sigmoid 函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 五、计算误差平均值
def cost(xMat, yMat, ws):
    # 根据逻辑回归的代价函数写的
    left = np.multiply(yMat, np.log(sigmoid(xMat * ws)))
    right = np.multiply(1 - yMat, np.log(1 - sigmoid(xMat * ws)))
    return np.sum(left + right) / -(len(xMat))


# 六、手写得到权值系数
def gradAscent(xArr, yArr):
    if scale == True:
        xArr = preprocessing.scale(xArr)   # 标准化
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    lr = 0.03
    epochs = 50000
    costList = []
    # 计算数据列数，有几列就有几个权值
    m, n = np.shape(xMat)
    # 初始化权值
    ws = np.mat(np.ones((n, 1)))

    for i in range(epochs + 1):
        # xMat和weights矩阵相乘
        h = sigmoid(xMat * ws)
        # 计算系数  根据求导公式写的
        ws_grad = xMat.T * (h - yMat) / m
        ws = ws - lr * ws_grad

        if i % 50 == 0:
            costList.append(cost(xMat, yMat, ws))
    return ws, costList

# 七、训练模型，得到权值和cost值的变化
def moder_demo():
    x_poly, y_data, poly_reg = poly_demo()
    ws, costList = gradAscent(x_poly, y_data)
    """print(ws)
    [[ 4.16787292]
     [ 2.72213524]
     [ 4.55120018]
     [-9.76109006]
     [-5.34880198]
     [-8.51458023]
     [-0.55950401]
     [-1.55418165]
     [-0.75929829]
     [-2.88573877]]
    """
    return ws, costList


# 八、作图
def zuotu_show():
    x_data, y_data, poly_reg = poly_demo()
    # 获取数据值所在的范围
    x_min, x_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    y_min, y_max = x_data[:, 2].min() - 1, x_data[:, 2].max() + 1

    # 生成网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    """
    np.r_按row来组合array，
    np.c_按colunm来组合array
    >>> a = np.array([1,2,3])
    >>> b = np.array([5,2,5])
    >>> np.r_[a,b]
    array([1, 2, 3, 5, 2, 5])
    >>> np.c_[a,b]
    array([[1, 5],
           [2, 2],
           [3, 5]])
    >>> np.c_[a,[0,0,0],b]
    array([[1, 0, 5],
           [2, 0, 2],
           [3, 0, 5]])
    """
    ws, costList = moder_demo()
    # ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
    z = sigmoid(poly_reg.fit_transform(np.c_[xx.ravel(), yy.ravel()]).dot(np.array(ws)))
    for i in range(len(z)):
        if z[i] > 0.5:
            z[i] = 1
        else:
            z[i] = 0
    z = z.reshape(xx.shape)

    # 等高线图
    cs = plt.contourf(xx, yy, z)
    date_show()
    plt.show()


# 九、预测
def predict(x_data, ws):
    # if scale == True:
    #   x_data = preprocessing.scale(x_data)
    xMat = np.mat(x_data)
    ws = np.mat(ws)
    return [1 if x >= 0.5 else 0 for x in sigmoid(xMat*ws)]


# 十、测试
def ceshi_demo():
    x_poly, y_data, poly_reg = poly_demo()
    ws, costList = gradAscent(x_poly, y_data)
    predictions = predict(x_poly, ws)

    print(classification_report(y_data, predictions))


if __name__ == '__main__':
    # 代码1：载入数据
    # get_data()

    # 代码2：散点图显示数据
    # date_show()

    # 代码3：特征处理, 使特征添加多项非线性特征
    # poly_demo()

    # 代码7：训练模型，得到权值和cost值的变化
    # moder_demo()

    # 代码8：作图
    # zuotu_show()

    # 代码10：测试
    ceshi_demo()
