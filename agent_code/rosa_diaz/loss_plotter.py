#!usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_list):
    t = np.arange(len(loss_list))
    l = np.array(loss_list)
    plt.plot(t, l)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()

loss = np.loadtxt("loss_log.txt")
plot_loss(loss)
