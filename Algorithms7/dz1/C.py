m = int(input())
costs = [2 ** i for i in range(31)]
# cost = [2, 1, 4, 4]
seconds = [int(x) for x in input().split()]


seconds_cost = [(seconds[i], costs[i]) for i in range(len(seconds))]

i = 0
j = 0
while True:
    repeats = int(m / seconds_cost[j][0]) + 1
    for _ in range(repeats):
        seconds_cost.insert(j, seconds_cost[j])
    j += repeats
    i += 1
    if i == len(seconds) - 1:
        break

list_of_costs = [0 if i == 0 else -1 for i in range(m+1)]

for second, cost in seconds_cost:

    for j in range(m - second, -1, -1):
        if list_of_costs[j] != -1:
            if list_of_costs[j + second] == -1 or list_of_costs[j + second] > cost + list_of_costs[j]:
                list_of_costs[j + second] = cost + list_of_costs[j]

print(list_of_costs[-1])

