import numpy as np

a11 = 0
a21 = 1

a12 = 0.5
a22 = 0.5

a13 = 0.2
a23 = 0.8

a14 = 0.7
a24 = 0.3

a15 = 0.9
a25 = 0.1

predicted = np.array([
    [a11, a21],
    [a12, a22],
    [a13, a23],
    [a14, a24],
    [a15, a25]
])

real = np.array([1, 0, 1, 0, 1])


# 损失函数
def precise_loss_function(predicted, real):
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(predicted * real_matrix, axis=1)
    return 1 - product


print(precise_loss_function(predicted, real))
