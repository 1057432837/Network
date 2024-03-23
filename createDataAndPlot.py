import numpy as np
import math
import random
import matplotlib.pyplot as plt

NUM_OF_DATA = 1000


def tag_entry(x, y):
    if x ** 2 + y ** 2 < 1:
        tag = 0
    else:
        tag = 1
    return tag


def creat_data(num_of_data):
    entry_list = []
    for i in range(num_of_data):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        tag = tag_entry(x, y)
        entry = [x, y, tag]
        entry_list.append(entry)
    return np.array(entry_list)


def plot_data(data, title):
    color = []
    for i in data[:, 2]:
        if i == 0:
            color.append("orange")
        else:
            color.append("blue")
    plt.scatter(data[:, 0], data[:, 1], c=color)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    data = creat_data(NUM_OF_DATA)
    print(data)
    plot_data(data, 'Demo')
