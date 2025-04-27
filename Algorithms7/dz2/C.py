n = int(input())
lst = list(map(int, input().split()))

j = 0
while n > 2 ** j:
  j += 1
for _ in range(2 ** j - n):
    lst.append(float('-inf'))
n = 2 ** j

def maximum(a, b):
    if a >= b:
        return a
    return b


class TreeNode:
    def __init__(self, son1, son2, val=0, idx=0):
        self.son1 = son1
        self.son2 = son2
        if self.son1 and self.son2:
            self.value = maximum(self.son1.value, self.son2.value)
            if self.son1.value >= self.son2.value:
                self.index = self.son1.index
            else:
                self.index = self.son2.index

        else:
            self.value = val
            self.index = idx


tree = []
for idx in range(j, -1, -1):
    if idx == j:
        for i in range(0, (2 ** idx), 1):
            tree.append(TreeNode(None, None, val=lst[i], idx=i))
            start_idx = 0
    else:
        for i in range(0, (2 ** idx) * 2, 2):
            tree.append(TreeNode(son1=tree[start_idx + i], son2=tree[start_idx + i + 1]))
        start_idx += 2 ** idx * 2


def megatron(L, R, node=tree[-1], L_tree=0, R_tree=n-1):
    if R_tree < L or L_tree > R:
        return float('-inf'), 0
    elif L <= L_tree and R_tree <= R:
        return node.value, node.index
    else:
        m1, i1 = megatron(L, R, node.son1, L_tree, (L_tree + R_tree) // 2)
        m2, i2 = megatron(L, R, node.son2, (L_tree + R_tree) // 2 + 1, R_tree)
        if m1 >= m2:
            return m1, i1
        else:
            return m2, i2

out = []
k = int(input())
for i in range(k):
    L, R = map(int, input().split())
    val , index = megatron(L-1, R-1)
    out.append((val, index+1))

for elem in out:
    print(*elem)

