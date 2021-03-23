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


def plot_reward(reward_list):
    t = np.arange(len(reward_list))
    l = np.array(reward_list)
    plt.plot(t, l)
    # plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.show()


loss = np.loadtxt("loss_log.txt")
reward = np.loadtxt("reward_log.txt")
plot_loss(loss)
plot_reward(reward)
