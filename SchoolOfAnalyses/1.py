def answer(n, k, lst):
    if k == 0:
        return 0
    max_unique = 0
    freq = {}
    unique = 0

    for i in range(n):
        if i >= k:
            left_element = lst[i - k]
            freq[left_element] -= 1
            if freq[left_element] == 0:
                unique -= 1
        new_element = lst[i]
        if new_element not in freq or freq[new_element] == 0:
            unique += 1
            freq[new_element] = 1
        else:
            freq[new_element] += 1
        if i >= k - 1:
            max_unique = max(max_unique, unique)
    return max_unique


n, k = map(int, input().split())
lst = list(map(int, input().split()))
print(answer(n, k, lst))