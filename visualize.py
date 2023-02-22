import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import pandas as pd
import numpy as np


def visualize_validation(filename: str, start: int, end: int):
    val = pd.read_csv(filename)

    ys = val["prices"].to_list()[start:end]
    choices = val["actions"].to_list()[start:end]

    sell_markers = [i for i, x in enumerate(choices) if x == 0]
    buy_markers = [i for i, x in enumerate(choices) if x == 2]
    hold_markers = [i for i, x in enumerate(choices) if x == 1]

    # plot the agent's behavior within a specific time segment: green = sell, red = sell
    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="lime", ls="dotted", markersize=10, markevery=sell_markers, label="sell electricity")
    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="grey", ls="dotted", markersize=10, markevery=hold_markers, label="sell electricity")
    plt.plot(list(range(len(ys))), ys, color="black", marker="o", mfc="orange", ls="dotted", markersize=10, markevery=buy_markers, label="buy electricity")
    plt.legend()
    plt.show()

    # plot change in cash
    plt.plot(val["cash"].to_list())
    plt.show()
    plt.plot(val["water_level"].to_list()[start:end])
    plt.show()


def visualize_v(qtable: np.array, drop_inds: tuple[int], x_axis_label, y_axis_label) -> None:
    q_av = np.average(qtable, axis=drop_inds)

    q1_ind = np.argmax(q_av, axis=-1)
    q1_val = np.amax(q_av, axis=-1)

    print(q1_ind)

    # Generate X, Y, and Z coordinates
    X, Y = np.meshgrid(range(q1_val.shape[0]), range(q1_val.shape[1]))
    Z = q1_val.transpose()
    
    norm = plt.Normalize(vmin=q1_ind.min().min(), vmax=q1_ind.max().max())

    # Plot the surface plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(norm(q1_ind.transpose())), shade=False)

    # Label the x-axis, y-axis, and z-axis
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_zlabel('V-Value')

    # Set the title of the plot
    ax.set_title(f'V-Values of Trained Agent\'s Policy Table for States \'{x_axis_label}\' and \'{y_axis_label}\'')

    col1, col2, col3 = cm.viridis(np.array([0.0, 0.6, 1.0]))

    # Add legend with proxy artists
    col1_patch = mpatches.Patch(color=col1, label='Sell (empty dam)')
    col2_patch = mpatches.Patch(color=col2, label='Hold (do nothing)')
    col3_patch = mpatches.Patch(color=col3, label='Buy (fill dam)')
    plt.legend(handles=[col1_patch, col2_patch, col3_patch], loc='center left')

    plt.show()


def visualize_progress(list_of_numbers: list, step=100):

    # Plot the numbers using a line plot
    plt.plot(range(0, len(list_of_numbers)*step, step), list_of_numbers)

    # Add a title and labels for the x and y axes
    plt.title("Validation Scores throughout Training")
    plt.xlabel("Simulation number")
    plt.ylabel("Final Validation Profit")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    visualize_validation("evaluated_model.csv", 15000, 15200)