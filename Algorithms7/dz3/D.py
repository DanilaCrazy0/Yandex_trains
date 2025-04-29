n = int(input())

size = n.bit_length()
maxi = n
for k in range(size):
    last_bit = n & 1
    a = (n >> 1) | (last_bit << (size-1))
    if a > maxi:
        maxi = a
    n = a

print(maxi)