n, m = list(map(int, input().split()))
lst_x = [int(x) for x in input().split()]
lst_y = [int(y) for y in input().split()]

lst_x = [(elem, i) for i, elem in enumerate(lst_x)]
lst_y = [(elem, i) for i, elem in enumerate(lst_y)]

sorted_x = sorted(lst_x, key=lambda x: (-x[0], x[1]))
sorted_y = sorted(lst_y, key=lambda x: (-x[0], x[1]))

list_of_pairs = []

count = 0
for i in range(len(sorted_x)):
    if sorted_x[i][0] + 1 <= sorted_y[i][0]:
        list_of_pairs.append((sorted_x[i][1], sorted_y[i][1]))
        count += 1

    else:
        list_of_pairs.append((sorted_x[i][1], -1))
        sorted_y.insert(i, (0, 0))

sort_pairs = sorted(list_of_pairs, key=lambda x: x[0])

print(count)
for elem in sort_pairs:
    if elem[1] == -1:
        print(0, end=' ')
    else:
        print(elem[1] + 1, end=' ')