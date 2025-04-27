import numpy as np
N, e, s = map(int, input().split())

if e - s > 0:
    e = e - s
else:
    e = N + e - s

s = 0
lst = [0] * (N-1)
print(e, s)

def dist(e):
    return min(e, N - e)

up_diag = [dist(i) / N for i in range(1, N-1)]
low_diag = up_diag[::-1]
print(up_diag)

m1, m2 = np.eye(N-1, k=1), np.eye(N-1, k=-1)
for i in range(m1.shape[0]):
    if i != m1.shape[0] - 1:
        m1[i] *= up_diag[i]
    if i != 0:
        m2[i] *= low_diag[i-1]

matrix = m1 + m2

print(matrix)
B = np.linalg.inv(np.eye(N-1) - matrix)
print(B)
print(np.sum(B, axis=1)[e-1])
