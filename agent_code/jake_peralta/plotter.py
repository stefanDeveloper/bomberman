import matplotlib.pyplot as plt
import numpy as np

LOG_NAME = "with_batch"
LOG_NAME_NEW = "without_batch"

def plot(coin_list, name, title, y_label, x_label, log=False):
    t = np.arange(len(coin_list))
    l = np.array(coin_list)
    plt.plot(t, l)
    if log:
        plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{name}.pdf")
    plt.show()


loss = np.loadtxt(f"loss_log_{LOG_NAME}.txt")
reward = np.loadtxt(f"reward_log_{LOG_NAME}.txt")
coins = np.loadtxt(f"coin_log_{LOG_NAME}.txt")

plot(loss, f"loss_log_{LOG_NAME}", "Batch 128", "Loss", "Steps", True)
plot(reward, f"reward_log_{LOG_NAME}", "Batch 128", "Reward", "Steps")
plot(coins, f"coin_log_{LOG_NAME}", "Batch 128", "Coins", "Rounds")

loss = np.loadtxt(f"loss_log_{LOG_NAME_NEW}.txt")
reward = np.loadtxt(f"reward_log_{LOG_NAME_NEW}.txt")
coins = np.loadtxt(f"coin_log_{LOG_NAME_NEW}.txt")

plot(loss, f"loss_log_{LOG_NAME_NEW}", "Batch 128", "Loss", "Steps", True)
plot(reward, f"reward_log_{LOG_NAME_NEW}", "Batch 128", "Reward", "Steps")
plot(coins, f"coin_log_{LOG_NAME_NEW}", "Batch 128", "Coins", "Rounds")
