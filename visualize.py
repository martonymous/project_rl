import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualize_validation(filename: str, start: int, end: int):
    val = pd.read_csv(filename)

    ys = val["prices"].to_list()[start:end]
    choices = val["actions"].to_list()[start:end]

    sell2_markers = [i for i, x in enumerate(choices) if x == 0]
    sell_markers = [i for i, x in enumerate(choices) if x == 1]
    hold_markers = [i for i, x in enumerate(choices) if x == 2]
    buy_markers = [i for i, x in enumerate(choices) if x == 3]
    buy2_markers = [i for i, x in enumerate(choices) if x == 4]

    # plot the agent's behavior within a specific time segment: green = sell, red = sell
    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="green", ls="dotted", markersize=10, markevery=sell2_markers, label="sell electricity")
    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="lime", ls="dotted", markersize=10, markevery=sell_markers, label="sell electricity")
    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="grey", ls="dotted", markersize=10, markevery=hold_markers, label="sell electricity")
    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="orange", ls="dotted", markersize=10, markevery=buy_markers, label="buy electricity")
    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="red", ls="dotted", markersize=10, markevery=buy2_markers, label="buy electricity")
    plt.legend()
    plt.show()

    # plot change in cash
    plt.plot(val["cash"].to_list())
    plt.show()
    plt.plot(val["water_level"].to_list()[start:end])
    plt.show()

if __name__ == "__main__":
    visualize_validation("data/eval_long.csv", 15000, 16500)