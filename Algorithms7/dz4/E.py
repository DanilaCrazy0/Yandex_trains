n = int(input())
a = list(map(int, input().split()))

left = [(i - 1) % n for i in range(n)]
right = [(i + 1) % n for i in range(n)]

removed = [-1] * n
current_round = []

for i in range(n):
    l = left[i]
    r = right[i]
    if a[i] < a[l] and a[i] < a[r]:
        current_round.append(i)

k = 1
n_remaining = n

while current_round and n_remaining > 2:
    to_remove = set()
    for i in current_round:
        if removed[i] != -1:
            continue
        l = left[i]
        r = right[i]
        if a[i] < a[l] and a[i] < a[r]:
            to_remove.add(i)
    if not to_remove:
        break
    for i in to_remove:
        removed[i] = k
    n_remaining -= len(to_remove)

    next_round = set()
    for i in to_remove:
        l = left[i]
        r = right[i]
        if l not in to_remove:
            right[l] = r
        if r not in to_remove:
            left[r] = l
        if l not in to_remove and removed[l] == -1:
            next_round.add(l)
        if r not in to_remove and removed[r] == -1:
            next_round.add(r)

    current_round = list(next_round)
    k += 1
    if n_remaining <= 2:
        break

result = [0 if x == -1 else x for x in removed]
print(' '.join(map(str, result)))