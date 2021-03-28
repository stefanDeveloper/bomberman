#!usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_alt(loss_list, num_points=1000):
    l = np.array(loss_list)
    max_entries = int(len(l)/num_points)
    g = np.mean(l[:max_entries*num_points].reshape(num_points, -1), axis=1) # cut off the last stuff so reshape works
    x = np.arange(len(g))
    plt.plot(x, g)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.show()
loss = np.loadtxt("loss_log.txt")
print('NOTE: This only works if you have more than "num_points" datapoints.')
plot_loss_alt(loss)
