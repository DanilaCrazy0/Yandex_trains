deque = []

while True:
    command = input().split()

    if not command:
        continue

    if command[0] == "push_back":
        n = int(command[1])
        deque.append(n)
        print("ok")

    if command[0] == "push_front":
        n = int(command[1])
        deque.insert(0, n)
        print("ok")

    elif command[0] == "pop_front":
        if len(deque) == 0:
            print("error")
        else:
            print(deque[0])
            del deque[0]

    elif command[0] == "pop_back":
        if len(deque) == 0:
            print("error")
        else:
            print(deque.pop())

    elif command[0] == "front":
        if len(deque) == 0:
            print("error")
        else:
            print(deque[0])

    elif command[0] == "back":
        if len(deque) == 0:
            print("error")
        else:
            print(deque[-1])

    elif command[0] == "size":
        print(len(deque))

    elif command[0] == "clear":
        deque.clear()
        print("ok")

    elif command[0] == "exit":
        print("bye")
        break