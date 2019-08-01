import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt

# 一、数据准备
def data_load():
    # 读入数据
    data = genfromtxt(r"longley.csv", delimiter=',')
    # print(data)

    # 切分数据
    x_data = data[1:, 2:]
    y_data = data[1:, 1]
    # print(x_data)
    # print(y_data)

    return x_data, y_data

# 二、创建模型及图示
def get_model(x_data, y_data):
    """

    :param x_data: 训练数据x
    :param y_data: 训练标签y
    :return: 模型model
    """
    # 生成从0.001到1递增的50（默认）个值,用于挑选出最合适的岭系数
    alphas_to_test = np.linspace(0.001, 1)
    # 创建模型，RidgeCV：交叉验证。store_cv_values：保存误差值
    model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
    model.fit(x_data, y_data)
    # 岭系数---model.alpha_:为最佳的岭系数
    print("最佳的岭系数:\n", model.alpha_)
    # loss值---上面保存的误差值形状为（16，50）16个样本分别作为验证集，50个岭系数
    print("模型保存误差值形状：\n", model.cv_values_.shape)
    # 画图
    # 岭系数跟loss值的关系（岭系数，16个样本作为验证集的平均误差）
    plt.plot(alphas_to_test, model.cv_values_.mean(axis=0))
    # 选取的岭系数值的位置（岭系数，最小的平均误差）
    plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)), 'ro')
    plt.show()

    return model



# 三、利用模型预测
def predict_demo(model, x_text):
    """
    利用模型预测
    :param model: 模型
    :param x_text: 测试数据
    :return: 预测值
    """
    predict = model.predict(x_text)
    print("预测值为：\n", predict)


if __name__ == '__main__':
    # 一、数据准备
    x_data, y_data = data_load()
    # 二、创建模型及图示
    model = get_model(x_data, y_data)
    # 三、利用模型预测
    x_text = x_data[2, np.newaxis]
    predict_demo(model, x_text)
    print("真实值为：\n", y_data[2])


