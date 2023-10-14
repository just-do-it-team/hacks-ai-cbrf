import random
import numpy as np
from numpy import linalg
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
t0 = time.time()

y_smp_train = np.load('F:/DATASETS/hacks-ai-2023/cbrf/train/y_smp_train.npy')
pars_smp_train = np.load('F:/DATASETS/hacks-ai-2023/cbrf/train/pars_smp_train.npy')

INPUT_DIM = 600
OUT_DIM = 15
H_DIM = 6000 ####
NUM_EPOCHS = 500
BATCH_SIZE = 150
LEARNING_RATE = 0.000002

W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)
W2 = np.random.rand(H_DIM, OUT_DIM)
b2 = np.random.rand(1, OUT_DIM)

W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)


def relu_deriv(t):
    return (t >= 0).astype(float)

def predict(x):
    t1 = x @ W1 + b1
    h1 = np.tanh(t1)
    t2 = h1 @ W2 + b2
    z = t2
    return z

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

loss_arr = []

dataset = [(y_smp_train[i].reshape(1,600), pars_smp_train[i].reshape(1,15)) for i in range(15000)]

for q in tqdm(quantiles):
    for ep in range(NUM_EPOCHS):
        # random.shuffle(dataset)
        for i in range(len(dataset) // BATCH_SIZE):
            batch_x, batch_y = zip(*dataset[i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE])
            x = np.concatenate(batch_x)
            y = np.array(batch_y).reshape(BATCH_SIZE, 15)

            # forward Прямое распространение
            t1 = x @ W1 + b1
            h1 = np.tanh(t1)
            t2 = h1 @ W2 + b2
            z = t2
            E = np.sum((q/100*((y-z)>0) + (1-q/100)*((y-z)<=0))*abs(y-z))

            # Backward Обратное распространение, вычисление градиента
            dE_dt2 = z - y
            dE_dW2 = h1.T @ dE_dt2
            dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
            dE_dh1 = dE_dt2 @ W2.T
            dE_dt1 = dE_dh1 * relu_deriv(t1)
            dE_dW1 = x.T @ dE_dt1
            dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

            W1 = W1 - LEARNING_RATE * dE_dW1
            b1 = b1 - LEARNING_RATE * dE_db1
            W2 = W2 - LEARNING_RATE * dE_dW2
            b2 = b2 - LEARNING_RATE * dE_db2

            loss_arr.append(E/BATCH_SIZE)
    np.save(f"W1-q{q}.npy", W1)
    np.save(f"b1-q{q}.npy", b1)
    np.save(f"W2-q{q}.npy", W2)
    np.save(f"b2-q{q}.npy", b2)


print(time.time() - t0, "sec END")

# График зависимости ошибки от итераций
plt.plot(loss_arr)
plt.show()
