import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    a = [0] * (N + 1)
    for i in range(1, N + 1):
        a[i] = int(sys.stdin.readline())

    visited = [False] * (N + 1)
    count = 0

    for i in range(1, N + 1):
        if not visited[i]:
            current = i
            path = []
            while True:
                if visited[current]:
                    if current in path:
                        idx = path.index(current)
                        cycle = path[idx:]
                        count += 1
                    break
                visited[current] = True
                path.append(current)
                current = a[current]
    print(count)

if __name__ == "__main__":
    main()