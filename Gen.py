import numpy as np

n = 30
m = 1000000
seed = 34

np.random.seed(seed)

X = np.random.randn(m, n)
X2 = np.random.randn(m, n)

W = np.random.randn(n)
b = np.random.randn()
Z = X @ W + b
Y = (Z > 0).astype(int)

Z2 = X2 @ W + b
Y2 = (Z2 > 0).astype(int)

print("Число примеров класса 0:", np.sum(Y == 0))
print("Число примеров класса 1:", np.sum(Y == 1))

np.savez("dataset.npz", X=X, Y=Y)
np.savez("test.npz", X=X2, Y=Y2)
