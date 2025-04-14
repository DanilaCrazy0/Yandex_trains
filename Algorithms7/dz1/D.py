n, m = list(map(int, input().split()))
masses = [int(x) for x in input().split()]
list_of_masses = [0 if i == 0 else -1 for i in range(m+1)]

for i in range(len(masses)):

    for j in range(m - masses[i], -1, -1):
        if list_of_masses[j] != -1:
            if list_of_masses[j + masses[i]] == -1 or list_of_masses[j + masses[i]] < masses[i] + list_of_masses[j]:
                list_of_masses[j + masses[i]] = masses[i] + list_of_masses[j]

print(max(list_of_masses))