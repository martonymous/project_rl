import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

val = pd.read_csv("data/val.csv")

ys = val["prices"].to_list()[:100]
choices = np.random.choice([0, 1, 2], size=len(ys), p=[.1, .1, .8]).tolist()

df = pd.DataFrame(choices)

sell_markers = [i for i, x in enumerate(choices) if x == 0]
buy_markers = [i for i, x in enumerate(choices) if x == 1]

plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="green", ls="dotted", markersize=10, markevery=sell_markers, label="sell electricity")
plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="red", ls="dotted", markersize=10, markevery=buy_markers, label="buy electricity")
plt.legend()
plt.show()