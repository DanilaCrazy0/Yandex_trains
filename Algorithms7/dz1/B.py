t = int(input())

lst_samples = []
for i in range(t):
    n = int(input())
    sample = [int(x) for x in input().split()]
    lst_samples.append(sample)

lst_intervals = []
global_max = 0
for sample in lst_samples:
    maxi = max(sample)
    if maxi > global_max:
        global_max = maxi

for sample in lst_samples:

    intervals = []
    interval = []
    maximum = global_max

    for i in range(len(sample)):
        elem = sample[i]
        if elem < maximum:
            maximum = elem

        if len(interval) + 1 <= elem and len(interval) + 1 <= maximum:
            interval.append(elem)
            if i == len(sample) - 1:
                intervals.append(len(interval))
                maximum = global_max

        elif len(interval) + 1 > elem or len(interval) + 1 > maximum:
            intervals.append(len(interval))
            if i == len(sample) - 1:
                interval = [elem]
                intervals.append(len(interval))
                maximum = global_max
            else:
                interval = [elem]
                maximum = elem

    print(len(intervals))
    print(*intervals)
