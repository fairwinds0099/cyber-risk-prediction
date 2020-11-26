import matplotlib.pyplot


class PlotUtils:

    @staticmethod
    def plot_data(X, y):
        plt = matplotlib.pyplot
        plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Label #0", alpha=1, linewidth=0.15)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Label #1", alpha=1, linewidth=0.15, color='red')
        plt.legend(bbox_to_anchor=(1.05, 1))
        return plt
