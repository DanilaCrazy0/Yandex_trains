import numpy as np


def solve():
    N, E, S = map(int, input().split())
    # Приводим к 0-based индексации
    E -= 1
    S -= 1

    # Функция для вычисления расстояния через вершину X
    def distance_via_X(current, X, target):
        # Расстояние от current до target через X
        # Т.е. путь current -> X -> ... -> target
        d1 = (X - current) % N
        d2 = (target - X) % N
        return d1 + d2

    # Непоглощающие состояния: все, кроме S
    states = [i for i in range(N) if i != S]
    num_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    Q = np.zeros((num_states, num_states))

    for i in range(num_states):
        current = states[i]
        # Для каждой возможной следующей вершины X (соседи current)
        # У крота есть 2 соседа (в многоугольнике)
        neighbors = [(current - 1) % N, (current + 1) % N]
        total_prob = 0
        for X in neighbors:
            if X == S:
                # Переход в поглощающее состояние, не добавляем в Q
                continue
            x = distance_via_X(current, X, S)
            prob = (N - x) / N
            # Нормировка, т.к. вероятности только для соседей
            total_prob += prob

        for X in neighbors:
            if X == S:
                continue
            x = distance_via_X(current, X, S)
            prob = (N - x) / N
            # Нормируем вероятности, чтобы сумма переходов была 1
            if total_prob > 0:
                norm_prob = prob / total_prob
            else:
                norm_prob = 0
            j = state_to_idx[X]
            Q[i, j] = norm_prob

    # Матрица I - Q
    I = np.eye(num_states)
    IQ = I - Q
    # Фундаментальная матрица N = (I - Q)^-1
    try:
        N_mat = np.linalg.inv(IQ)
    except np.linalg.LinAlgError:
        # Если матрица вырожденная (например, если нет непоглощающих состояний)
        N_mat = np.zeros((num_states, num_states))

    # Мат. ожидание — сумма строки начального состояния
    if E == S:
        print(0)
        return
    initial_idx = state_to_idx[E]
    expectation = np.sum(N_mat[initial_idx, :])
    # Добавляем 1, так как каждый шаг — это день
    expectation += 1
    print(expectation / 2)


solve()