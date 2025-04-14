n, m = list(map(int, input().split()))
masses = list(map(int, input().split()))
costs = list(map(int, input().split()))
masses_cost = [(masses[i], costs[i]) for i in range(len(masses))]


list_of_costs = [0 if i == 0 else -1 for i in range(m+1)]

for second, cost in masses_cost:

    for j in range(m - second, -1, -1):
        if list_of_costs[j] != -1:
            if list_of_costs[j + second] == -1 or list_of_costs[j + second] < cost + list_of_costs[j]:
                list_of_costs[j + second] = cost + list_of_costs[j]

print(max(list_of_costs))

