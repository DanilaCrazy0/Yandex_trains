N, K = map(int, input().split())

if K == 0:
    print("NO")
    print(1, 1, 1)
else:
    x_coords = set()
    y_coords = set()
    z_coords = set()

    for _ in range(K):
        x, y, z = map(int, input().split())
        x_coords.add(x)
        y_coords.add(y)
        z_coords.add(z)

    all_covered = True
    for i in range(1, N + 1):
        if i not in x_coords or i not in y_coords or i not in z_coords:
            all_covered = False
            break

    if all_covered:
        print("YES")
    else:
        x_missing = None
        y_missing = None
        z_missing = None

        for i in range(1, N + 1):
            if i not in x_coords:
                x_missing = i
                break
        for i in range(1, N + 1):
            if i not in y_coords:
                y_missing = i
                break
        for i in range(1, N + 1):
            if i not in z_coords:
                z_missing = i
                break

        x = x_missing if x_missing is not None else 1
        y = y_missing if y_missing is not None else 1
        z = z_missing if z_missing is not None else 1

        if x == 1 and 1 in x_coords:
            for i in range(1, N + 1):
                if i not in x_coords:
                    x = i
                    break
        if y == 1 and 1 in y_coords:
            for i in range(1, N + 1):
                if i not in y_coords:
                    y = i
                    break
        if z == 1 and 1 in z_coords:
            for i in range(1, N + 1):
                if i not in z_coords:
                    z = i
                    break

        print("NO")
        print(x, y, z)