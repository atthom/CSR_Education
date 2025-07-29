from random import randint, random


table = None


def init(n_state, n_action):
    global table
    table = [None] * n_state
    for i in range(n_state):
        table[i] = [0] * n_action
    print('init table ->', table)


def take_decision(state):
    n_state, n_action = len(table), 1 #len(table[0])
    exploration = 0.1
    r = random()
    if r > exploration:
        line = table[state]
        action, m = 0, line[0]
        for i in range(n_action):
            if line[i] > m:
                action, m = i, line[i]
    else:
        action = randint(0, n_action-1)

    return action


def learn(old_state, action, reward, new_state):
    alpha = .2
    gamma = .8
    old_q = table[old_state][action]
    new_q = (1 - gamma) * reward + gamma * max(table[new_state])
    table[old_state][action] = old_q + alpha * (new_q - old_q)
    print('update ->', table)
