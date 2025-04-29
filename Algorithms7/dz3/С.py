import sys
from collections import deque

class Dinic:
    def __init__(self, N):
        self.N = N
        self.g = [[] for _ in range(N)]
    def add_edge(self, u, v, cap):
        self.g[u].append([v, cap, len(self.g[v])])
        self.g[v].append([u,  0,  len(self.g[u])-1])
    def max_flow(self, s, t):
        flow = 0
        N = self.N
        while True:
            # BFS level graph
            level = [-1]*N
            q = deque([s]); level[s]=0
            while q:
                u = q.popleft()
                for v,cap,rev in self.g[u]:
                    if cap and level[v]<0:
                        level[v] = level[u]+1
                        q.append(v)
            if level[t]<0:
                return flow
            it = [0]*N
            # DFS blocking flow
            def dfs(u, f):
                if u==t: return f
                for i in range(it[u], len(self.g[u])):
                    v,cap,rev = self.g[u][i]
                    if cap and level[v]==level[u]+1:
                        pushed = dfs(v, min(f, cap))
                        if pushed:
                            self.g[u][i][1] -= pushed
                            self.g[v][rev][1] += pushed
                            return pushed
                    it[u] += 1
                return 0
            while True:
                pushed = dfs(s, 10**18)
                if not pushed: break
                flow += pushed

def solve():
    input = sys.stdin.readline
    n = int(input())
    A = list(map(int, input().split()))
    C = [bin(a).count('1') for a in A]
    L = max(a.bit_length() for a in A)
    S = sum(C)
    if S % 2 == 1:
        print("impossible")
        return

    # 2) pick column‐degrees s_j, all even
    cap = n if n % 2 == 0 else n - 1
    s = [0]*L
    S_rem = S
    for j in range(L):
        take = min(cap, S_rem)
        s[j] = take
        S_rem -= take

    SRC = 0
    SNK = n + L + 1
    flow = Dinic(SNK+1)
    for i in range(n):
        flow.add_edge(SRC, i+1, C[i])
    for i in range(n):
        for j in range(L):
            flow.add_edge(i+1, n+1+j, 1)
    for j in range(L):
        flow.add_edge(n+1+j, SNK, s[j])

    f = flow.max_flow(SRC, SNK)
    if f != S:
        print("impossible")
        return

    B = [0]*n
    for i in range(n):
        for edge in flow.g[i+1]:
            v, cap_left, rev = edge
            if n+1 <= v <= n+L and cap_left == 0:
                bitpos = v - (n+1)
                B[i] |= (1 << bitpos)
    print(*B)

solve()
