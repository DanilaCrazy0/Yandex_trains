x = int(input())

bits = x.bit_length()
c = 0
for k in range(bits):
    if x & (1 << k):
        c += 1
print(c)