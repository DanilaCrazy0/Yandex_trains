stack = []

while True:
    command = input().split()

    if not command:
        continue

    if command[0] == "push":
        n = int(command[1])
        stack.append(n)
        print("ok")

    elif command[0] == "pop":
        if len(stack) == 0:
            print("error")
        else:
            print(stack.pop())

    elif command[0] == "back":
        if len(stack) == 0:
            print("error")
        else:
            print(stack[-1])

    elif command[0] == "size":
        print(len(stack))

    elif command[0] == "clear":
        stack.clear()
        print("ok")

    elif command[0] == "exit":
        print("bye")
        break