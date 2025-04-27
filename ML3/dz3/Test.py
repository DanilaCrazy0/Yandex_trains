import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    arr = list(map(int, input[ptr:ptr + N]))
    ptr += N

    # Дополняем массив до степени двойки
    size = 1
    while size < N:
        size <<= 1
    tree_max = [ -float('inf') ] * (2 * size)
    tree_cnt = [ 0 ] * (2 * size)

    # Заполняем листья
    for i in range(N):
        tree_max[size + i] = arr[i]
        tree_cnt[size + i] = 1
    for i in range(N, size):
        tree_max[size + i] = -float('inf')
        tree_cnt[size + i] = 0

    # Построение дерева
    for i in range(size - 1, 0, -1):
        left_max = tree_max[2 * i]
        right_max = tree_max[2 * i + 1]
        if left_max > right_max:
            tree_max[i] = left_max
            tree_cnt[i] = tree_cnt[2 * i]
        elif left_max < right_max:
            tree_max[i] = right_max
            tree_cnt[i] = tree_cnt[2 * i + 1]
        else:
            tree_max[i] = left_max
            tree_cnt[i] = tree_cnt[2 * i] + tree_cnt[2 * i + 1]

    def query(l, r):
        l += size
        r += size
        res_max = -float('inf')
        res_cnt = 0
        while l <= r:
            if l % 2 == 1:
                if tree_max[l] > res_max:
                    res_max = tree_max[l]
                    res_cnt = tree_cnt[l]
                elif tree_max[l] == res_max:
                    res_cnt += tree_cnt[l]
                l += 1
            if r % 2 == 0:
                if tree_max[r] > res_max:
                    res_max = tree_max[r]
                    res_cnt = tree_cnt[r]
                elif tree_max[r] == res_max:
                    res_cnt += tree_cnt[r]
                r -= 1
            l //= 2
            r //= 2
        return res_max, res_cnt

    K = int(input[ptr])
    ptr += 1
    output = []
    for _ in range(K):
        L = int(input[ptr]) - 1  # Переводим в 0-индексацию
        R = int(input[ptr + 1]) - 1
        ptr += 2
        max_val, cnt = query(L, R)
        output.append(f"{max_val} {cnt}")

    print('\n'.join(output))

if __name__ == "__main__":
    main()