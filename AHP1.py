import numpy as np


def ConsisTest(X):
    # 计算权重
    X = np.array(X)
    sum_X = X.sum(axis=0)
    (n, n) = X.shape
    sum_X = np.tile(sum_X, (n, 1))
    stand_X = X / sum_X

    # 算术平均法
    sum_row = stand_X.sum(axis=1)
    arithmetic_weights = sum_row / n
    print("算数平均法求权重的结果为：")
    print(arithmetic_weights)

    # 特征值法
    V, E = np.linalg.eig(X)
    max_value = np.max(V)
    print("最大特征值是：", max_value)
    max_v_index = np.argmax(V)
    max_eiv = E[:, max_v_index]
    eigen_weights = max_eiv / max_eiv.sum()
    print("特征值法求权重的结果为：")
    print(eigen_weights.real)  # 取实部
    print("———————————————————————————————")

    # 一致性检验
    CI = (max_value - n) / (n - 1)
    RI = np.array([0, 0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59])
    print('CI =', CI, 'RI =', RI[n])
    CR = CI / RI[n]
    if CR < 0.1:
        print("CR =", CR, "，小于0.1，通过一致性检验")
    else:
        print("CR =", CR, "，大于等于0.1，没有通过一致性检验，请修改判断矩阵")

    return arithmetic_weights, eigen_weights.real


# 根据图片中的矩阵构建判断矩阵
A = [
    [1, 3, 4, 5, 7, 7],  # 可用性
    [1 / 3, 1, 2, 2, 5, 4],  # 好用性
    [1 / 4, 1 / 2, 1, 1, 3, 3],  # 成本
    [1 / 5, 1 / 2, 1, 1, 3, 2],  # 考试功能
    [1 / 7, 1 / 5, 1 / 3, 1 / 3, 1, 1],  # 个性化服务
    [1 / 7, 1 / 4, 1 / 3, 1 / 2, 1, 1]  # 专业度
]

# 因素名称
factors = ["可用性", "好用性", "成本", "考试功能", "个性化服务", "专业度"]

print("=== 图片中的判断矩阵分析结果 ===")
print("判断矩阵：")
for i in range(len(A)):
    print(f"{factors[i]:<8}", A[i])

print("\n" + "=" * 50)
weights_arithmetic, weights_eigen = ConsisTest(A)

# 显示最终的权重排序
print("\n=== 最终权重排序（特征值法）===")
sorted_weights = sorted(zip(factors, weights_eigen), key=lambda x: x[1], reverse=True)
for i, (factor, weight) in enumerate(sorted_weights, 1):
    print(f"{i:2d}. {factor:<8}: {weight:.4f} ({weight * 100:.2f}%)")