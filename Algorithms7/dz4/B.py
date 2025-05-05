queue = []

while True:
    command = input().split()

    if not command:
        continue

    if command[0] == "push":
        n = int(command[1])
        queue.append(n)
        print("ok")

    elif command[0] == "pop":
        if len(queue) == 0:
            print("error")
        else:
            print(queue[0])
            del queue[0]

    elif command[0] == "front":
        if len(queue) == 0:
            print("error")
        else:
            print(queue[0])

    elif command[0] == "size":
        print(len(queue))

    elif command[0] == "clear":
        queue.clear()
        print("ok")

    elif command[0] == "exit":
        print("bye")
        break