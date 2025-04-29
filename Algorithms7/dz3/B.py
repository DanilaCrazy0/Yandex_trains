n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]

a = [0] * n

for i in range(n):
    for j in range(n):
        if i != j:
            a[i] |= matrix[i][j]

print(' '.join(map(str, a)))