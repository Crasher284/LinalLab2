import numpy as np
import time
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activate(X, W, vSigmoid):
    T = X
    T = T @ W
    T = vSigmoid(T)
    return T

def crossEntropy(Y, Yn, m):
    return -(1/m)*np.sum(Y * np.log(Yn+ 1e-10) + (1 - Y) * np.log(1 - Yn+ 1e-10))

def updateWs(W, X, Y, Yn, m, alpha):
    dJdW = (1/m)*X.T @ (Yn - Y)
    W=W-dJdW*alpha
    return W

alpha = 0.1
m = 1000000
n = 30
acc = 0.05
batch_size = 1000
vSigmoid = np.vectorize(sigmoid)

data = np.load('dataset.npz')
X = data['X']
X = np.c_[X, np.ones(X.shape[0])]
Y = data['Y']
W = np.random.randn(n+1)

ids = np.arange(m)
max_epochs = 100

t = time.time()
Yn = activate(X, W, vSigmoid)
J = crossEntropy(Y, Yn, m)
ers = []
ers.append(J)
print("Starting loss:", J)

epoch=0
while J>=acc or epoch>max_epochs:
    epoch+=1
    np.random.shuffle(ids)
    for i in range(0, m, batch_size):
        batch = ids[i:i + batch_size]
        X_b = X[batch]
        Y_b = Y[batch]
        Yn_b = activate(X_b, W, vSigmoid)
        W = updateWs(W, X_b, Y_b, Yn_b, batch_size, alpha)
    Yn = activate(X, W, vSigmoid)
    J = crossEntropy(Y, Yn, m)
    print(f"Epoch {epoch}, Loss: {J}")
    ers.append(J)
s = time.time() - t
print(f"Finished on epoch {epoch}, Loss: {J}")
print(f"Took {s:.2f} seconds to finish.")

data = np.load('test.npz')
X = data['X']
X = np.c_[X, np.ones(X.shape[0])]
Y = data['Y']

Y_pred = (activate(X, W, vSigmoid) > 0.5).astype(int)
accuracy = np.mean(Y_pred == Y)
print(f"Accuracy on tests: {accuracy}")

plt.figure('Loss Graph', figsize=(10, 6))
plt.plot(range(len(ers)), ers, label='Cross-Entropy Loss', marker='o')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (J)')
plt.title('Loss Decrease During Training')
plt.grid(True)
plt.legend()
plt.show()
    
    
