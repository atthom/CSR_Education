
import math


def init(n_input, n_output):
    """
    Perform any needed initialization of the supervised algorithm (i.e.
    create a neural network)
    :param n_input: number of inputs
    :param n_output: number of possible categories for output
    """
    pass


train_sensors = train_decisions = None


def learn(X_train, y_train):
    global train_sensors, train_decisions
    train_sensors, train_decisions = X_train, y_train
    loss = 0
    return loss


def take_decision(sensors):
    """
    Compute the label of a new data point
    :param sensors: data point
    :return: computed label of the new data point
    """
    if train_sensors is None:
        return 0
    return nearest_neighbor_decision(train_sensors, train_decisions, sensors)


def distance(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)


def all_distances(a, train_sensors):
    return [distance(a, b) for b in train_sensors]


def find_minimum(dlist):
    value_min = math.inf
    index_min = None
    for i in range(len(dlist)):
        if dlist[i] < value_min:
            value_min = dlist[i]
            index_min = i
    return index_min


def nearest_neighbor_decision(train_sensors, train_decisions, a):
    dlist = all_distances(a, train_sensors)
    idx_min = find_minimum(dlist)
    return train_decisions[idx_min]


# TEST ZONE

a = [0, 0]
b = [1, 2]
print("distance", distance(a, b))

# compute all distances
a = [0.4, 0.6]
train_sensors = [[0, 0], [0, 1], [1, 0]]
distance_list = all_distances(a, train_sensors)
print('distances to data', distance_list)

# minimum in a list
idx_min = find_minimum(distance_list)
print('index of minimum', idx_min)

# ALGO_KNN (with K=1 !)
a = [0.4, 0.6]
train_sensors = [[0, 0], [0, 1], [1, 0]]
train_decisions = [1, 2, 0]
decision = nearest_neighbor_decision(train_sensors, train_decisions, a)
print('ALGO_KNN', decision)


