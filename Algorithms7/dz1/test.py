from copy import deepcopy

n, m = list(map(int, input().split()))
masses = list(map(int, input().split()))
costs = list(map(int, input().split()))
masses_cost = [(masses[i], costs[i]) for i in range(len(masses))]


list_idx_costs = [[0, 0] if i == 0 else [-1, -1] for i in range(m+1)]
real_big = []

for i in range(len(masses_cost)):

    for j in range(m - masses_cost[i][0], -1, -1):
        if list_idx_costs[j] != [-1, -1]:
            if list_idx_costs[j + masses_cost[i][0]] == [-1, -1] or list_idx_costs[j + masses_cost[i][0]][1] < masses_cost[i][1] + list_idx_costs[j][1]:
                list_idx_costs[j + masses_cost[i][0]][1] = masses_cost[i][1] + list_idx_costs[j][1]
                list_idx_costs[j + masses_cost[i][0]][0] = i + 1
    real_big.append(deepcopy(list_idx_costs))

order = []

for i in range(len(real_big) - 1, 0, -1):
    if i == len(real_big) - 1:
        max_cost = 0
        idx = 0
        for j in range(len(real_big[i])):
            maxi = real_big[i][j][1]
            if maxi > max_cost:
                max_cost = maxi
                idx = j
        order.append(real_big[i][idx][0])
        next_mass = idx - masses_cost[real_big[i][idx][0] - 1][0]
    else:
        order.append(real_big[i][next_mass][0])
        next_mass -= masses_cost[real_big[i][next_mass][0] - 1][0]


order = sorted(order)
for elem in order:
    print(elem)