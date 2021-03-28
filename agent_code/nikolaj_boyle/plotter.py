import matplotlib.pyplot as plt
import numpy as np


def plot_loss(loss_list, stepSize):
    length = len(loss_list)
    lengthUse = length - (length % stepSize)
    t = np.arange(lengthUse, step=stepSize)
    l = np.array(loss_list)
    l = np.mean(l[:lengthUse].reshape(-1, stepSize), axis=1)
    plt.plot(t, l)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


def plot_reward(reward_list, stepSize):
    length = len(reward_list)
    lengthUse = length - (length % stepSize)
    t = np.arange(lengthUse, step=stepSize)
    l = np.array(reward_list)
    l = np.mean(l[:lengthUse].reshape(-1, stepSize), axis=1)
    plt.plot(t, l)
    # plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.show()

def plot_coins(coin_list, stepSize):
    length = len(coin_list)
    lengthUse = length - (length % stepSize)
    t = np.arange(lengthUse, step=stepSize)
    l = np.array(coin_list)
    l = np.mean(l[:lengthUse].reshape(-1, stepSize), axis=1)
    plt.plot(t, l)
    # plt.yscale("log")
    plt.xlabel("round")
    plt.ylabel("coins/round")
    plt.show()

def plot_coins_with_epsilon(eps_list, coins_list, stepSize):
    length = len(eps_list)
    lengthUse = length - (length % stepSize)
    t = np.arange(lengthUse, step=stepSize)
    l = np.array(eps_list)
    l = np.mean(l[:lengthUse].reshape(-1, stepSize), axis=1)
    plt.plot(t, l, label='Epsilon')
    length = len(coins_list)
    lengthUse = length - (length % stepSize)
    t = np.arange(lengthUse, step=stepSize)
    l = np.array(coins_list)
    l = np.mean(l[:lengthUse].reshape(-1, stepSize), axis=1)
    plt.plot(t, l, label='Coins/round')
    # plt.yscale("log")
    plt.xlabel("round")
    #plt.ylabel("coins/round")
    plt.legend()
    plt.show()


#loss = np.loadtxt("loss_log.txt")
reward = np.loadtxt("reward_log.txt")
coins = np.loadtxt("coin_log.txt")
eps = np.loadtxt("eps_log.txt")
step = 1000
#plot_loss(loss, step)
plot_reward(reward, step)
plot_coins(coins, 10)
plot_coins_with_epsilon(eps, coins, 10)
