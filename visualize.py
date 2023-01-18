import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualize_validation(filename: str, start: int, end: int):
    val = pd.read_csv(filename)

    ys = val["prices"].to_list()[start:end]
    choices = val["actions"].to_list()[start:end]

    sell_markers = [i for i, x in enumerate(choices) if x == 0]
    buy_markers = [i for i, x in enumerate(choices) if x == 1]

    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="green", ls="dotted", markersize=10, markevery=sell_markers, label="sell electricity")
    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="red", ls="dotted", markersize=10, markevery=buy_markers, label="buy electricity")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualize_validation("data/eval.csv", 1000, 1250)