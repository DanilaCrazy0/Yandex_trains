n, m = map(int, input().split())
masses = list(map(int, input().split()))
costs = list(map(int, input().split()))

# Инициализация DP-таблицы: dp[weight] = (max_cost, list_of_items)
dp = [(-1, []) for _ in range(m + 1)]
dp[0] = (0, [])

for i in range(n):
    mass = masses[i]
    cost = costs[i]
    for w in range(m, mass - 1, -1):
        if dp[w - mass][0] != -1:
            if dp[w][0] < dp[w - mass][0] + cost:
                dp[w] = (dp[w - mass][0] + cost, dp[w - mass][1] + [i + 1])

# Находим максимальную стоимость
max_cost = max(item[0] for item in dp)
# Находим все возможные комбинации с максимальной стоимостью и выбираем любую
for item in dp:
    if item[0] == max_cost:
        print('\n'.join(map(str, sorted(item[1]))))
        break