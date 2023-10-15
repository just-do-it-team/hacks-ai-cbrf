import numpy as np
from tqdm import tqdm
import time
t0 = time.time()


def predict(x):
    t1 = x @ W1 + b1
    h1 = np.tanh(t1)
    t2 = h1 @ W2 + b2
    z = t2
    return z

y_smp_test = np.load('y_smp_test.npy')
names = ('', '-q0.1', '-q0.25', '-q0.5', '-q0.75', '-q0.9')

dataset = [y_smp_test[i].reshape(1,600) for i in range(100000)]
# print(dataset)
answer = np.array([], ndmin=2)

for n in range(len(names)):
    W1 = np.load(f'W1{names[n]}.npy')
    b1 = np.load(f'b1{names[n]}.npy')
    W2 = np.load(f'W2{names[n]}.npy')
    b2 = np.load(f'b2{names[n]}.npy')
    sample = np.array([], ndmin=3)
    for x in tqdm(dataset):
        z = predict(x).reshape(1, 15, 1)
        if sample.size == 0:
            sample = z
        else:
            sample = np.concatenate((sample, z), axis=0)

    if answer.size == 0:
        answer = sample
    else:
        answer = np.concatenate((answer, sample), axis=2)
print(answer)
print(answer.shape)
np.save("submit2.npy", answer)