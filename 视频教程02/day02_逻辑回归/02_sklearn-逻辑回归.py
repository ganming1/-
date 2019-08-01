import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import linear_model

# 数据是否需要标准化
scale = False

# 一、载入数据
def get_data():
    data = np.genfromtxt("LR-testSet.csv", delimiter=",")
    x_data = data[:, :-1]
    y_data = data[:, -1]

    return x_data, y_data

# 二、散点图显示数据
def date_show():
    x_data, y_data = get_data()
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    # 切分不同类别的数据
    for i in range(len(x_data)):
        if y_data[i] == 0:
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


# 三、数据处理，添加偏置项
def data_process():
    data = np.genfromtxt("LR-testSet.csv", delimiter=",")
    x_data = data[:, :-1]
    y_data = data[:, -1, np.newaxis]

    # print(np.mat(x_data).shape)
    # print(np.mat(y_data).shape)

    # 给样本添加偏置项
    X_data = np.concatenate((np.ones((100, 1)), x_data), axis=1)
    # print(X_data.shape)
    return X_data, y_data


# 四、生成模型
def moder_demo():
    x_data, y_data = get_data()
    logistic = linear_model.LogisticRegression()
    logistic.fit(x_data, y_data)
    print("模型的参数：\n", logistic.coef_)   # 二维
    print("模型的偏置：\n", logistic.intercept_)
    return logistic

# 五、画图
def show_demo():
    logistic = moder_demo()
    if scale == False:
        # 画图决策边界
        date_show()
        x_test = np.array([[-4], [3]])
        y_test = (-logistic.intercept_ - x_test * logistic.coef_[0][0]) / logistic.coef_[0][1]
        plt.plot(x_test, y_test, 'k')
        plt.show()


# 六、预测评估
def predict_demo():
    x_data, y_data = get_data()
    logistic = moder_demo()
    predictions = logistic.predict(x_data)

    print(classification_report(y_data,predictions))

if __name__ == '__main__':
    # 代码1：

    # 代码2：


    # 代码4：生成模型
    # moder_demo()

    # 代码5：画图
    # show_demo()

    # 代码5：预测评估
    predict_demo()
