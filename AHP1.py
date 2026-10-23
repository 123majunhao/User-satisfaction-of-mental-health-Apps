import numpy as np


def ConsisTest(X):
    X = np.array(X, dtype=float)
    sum_X = X.sum(axis=0)
    n = X.shape[0]
    sum_X = np.tile(sum_X, (n, 1))
    stand_X = X / sum_X

    sum_row = stand_X.sum(axis=1)
    arithmetic_weights = sum_row / n
    print("算术平均法求权重的结果为：")
    print(arithmetic_weights)

    V, E = np.linalg.eig(X)
    max_v_index = np.argmax(V.real)
    max_value = V[max_v_index].real
    print("最大特征值是：", max_value)

    max_eiv = E[:, max_v_index].real
    eigen_weights = max_eiv / max_eiv.sum()
    print("特征向量法求权重的结果为：")
    print(eigen_weights)
    print("———————————————————————————————")

    RI = np.array([
        0, 0, 0, 0.52, 0.89, 1.12, 1.24, 1.36,
        1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59
    ])

    if n <= 2:
        CI = 0.0
        CR = 0.0
    else:
        CI = (max_value - n) / (n - 1)
        CR = CI / RI[n]

    print("CI =", CI, "RI =", RI[n])

    if CR < 0.1:
        print("CR =", CR, "，小于0.1，通过一致性检验")
    else:
        print("CR =", CR, "，大于等于0.1，未通过一致性检验")

    return arithmetic_weights, eigen_weights


A_china = [
    [1,     9, 5,     9, 9, 9],
    [1 / 9, 1, 1 / 4, 1, 1, 1],
    [1 / 5, 4, 1,     5, 5, 4],
    [1 / 9, 1, 1 / 5, 1, 1, 1],
    [1 / 9, 1, 1 / 5, 1, 1, 1],
    [1 / 9, 1, 1 / 4, 1, 1, 1]
]

A_us = [
    [1,     8, 3,     5,     8, 7],
    [1 / 8, 1, 1 / 6, 1 / 4, 1, 1],
    [1 / 3, 6, 1,     3,     6, 5],
    [1 / 5, 4, 1 / 3, 1,     3, 3],
    [1 / 8, 1, 1 / 6, 1 / 3, 1, 1],
    [1 / 7, 1, 1 / 5, 1 / 3, 1, 1]
]

A = A_china

factors = [
    "心理服务功能",
    "软件有效性",
    "软件易用性",
    "个性化服务",
    "社会支持网络",
    "隐私与安全"
]

print("=== 判断矩阵分析结果 ===")
print("判断矩阵：")
for i in range(len(A)):
    print(f"{factors[i]:<8}", A[i])

print("\n" + "=" * 50)
weights_arithmetic, weights_eigen = ConsisTest(A)

print("\n=== 最终权重排序（特征向量法）===")
sorted_weights = sorted(
    zip(factors, weights_eigen),
    key=lambda x: x[1],
    reverse=True
)

for i, (factor, weight) in enumerate(sorted_weights, 1):
    print(f"{i:2d}. {factor:<8}: {weight:.4f} ({weight * 100:.2f}%)")
