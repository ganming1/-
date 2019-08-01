import numpy as np
from numpy import genfromtxt
from sklearn import linear_model

# 读入数据
data = genfromtxt(r"longley.csv", delimiter=',')
# print(data)

# 切分数据
x_data = data[1:, 2:]
y_data = data[1:, 1]
print(x_data)
print(y_data)

# 创建模型
model = linear_model.LassoCV()
model.fit(x_data, y_data)

# lasso系数,系数前面的惩罚参数
print("LOSSO模型求得的lasso系数：\n", model.alpha_)
# 相关系数
print("LOSSO模型求得的相关系数：\n", model.coef_)

result = model.predict(x_data[-2, np.newaxis])
print("预测结果：\n", result)