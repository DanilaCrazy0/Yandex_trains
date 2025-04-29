

x_ax = 0
y_ax = 0
z_ax = 0

for _ in range(K):
    x, y, z = map(int, input().split())
    x_ax |= x
    y_ax |= y
    z_ax |= z

x_i, y_i, z_i = -1, -1, -1
for i in range(N):
    if not (x_ax & (1 << i)):
        x_i = i
        break

for j in range(N):
    if not (y_ax & (1 << j)):
        y_i = j
        break

for k in range(N):
    if not (z_ax & (1 << k)):
        z_i = k
        break

if x_i >= 0:
    print('NO')
    print(x_i + 1, N, N)

elif y_i >= 0:
    print('NO')
    print(N, y_i + 1, N)

elif z_i >= 0:
    print('NO')
    print(N, N, z_i + 1)

else:
    print('YES')