# /// script
# dependencies = ["scikit-learn", "matplotlib"]
# ///
"""
Pictures of what the two training scripts do.

    uv run visualize.py        (writes plot_regression.png and plot_iris.png next to this file)

Left to right, the story is: data -> what the model learned -> how it did on unseen data.
"""
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ---- Shared look: a quiet chart surface so the data is the loudest thing on the page.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # one colour per category, fixed order
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
})


# =====================================================================================
# Figure 1: the hours-studied line from train_stdlib.py
# =====================================================================================
data = [(1, 52), (2, 58), (3, 64), (4, 71), (5, 75), (6, 82), (7, 88), (8, 93)]
train, test = data[:-2], data[-2:]

weight, bias, learning_rate = 0.0, 0.0, 0.01
errors = []                                                # training error after each step
for step in range(5000):
    grad_w = grad_b = mse = 0.0
    for hours, score in train:
        error = weight * hours + bias - score
        grad_w += 2 * error * hours / len(train)
        grad_b += 2 * error / len(train)
        mse += error ** 2 / len(train)
    errors.append(mse)
    weight -= learning_rate * grad_w
    bias -= learning_rate * grad_b

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
fig.suptitle("train_stdlib.py: fitting a straight line to hours studied vs exam score",
             color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left")

# Left: the data and the line the model found.
xs = [0, 9.5]
left.plot(xs, [weight * x + bias for x in xs], color=BLUE, linewidth=2, zorder=2, label="Learned line")
left.scatter(*zip(*train), s=56, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Training rows (used to fit)")
left.scatter(*zip(*test), s=56, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Test rows (never seen)")
left.text(0.3, 101, f"score = {weight:.2f} x hours + {bias:.2f}", color=INK2, fontsize=9, va="top")
left.set(xlabel="Hours studied", ylabel="Exam score", xlim=(0, 9.5), ylim=(40, 105))
left.set_title("The learned line: unseen test rows land on it too", loc="left")
left.legend(loc="lower right")

# Right: the error shrinking while training. This curve IS "training".
right.plot(range(1, len(errors) + 1), errors, color=BLUE, linewidth=2)
right.set(xlabel="Training step", ylabel="Training error (mean squared, log scale)", yscale="log", xlim=(1, 5000))
right.set_yticks([0.1, 1, 10, 100, 1000, 10000], ["0.1", "1", "10", "100", "1,000", "10,000"])
right.tick_params(axis="y", which="minor", left=False)
right.set_title("The error shrinks, then flattens: 5000 steps is plenty", loc="left")
done = next(i for i, e in enumerate(errors) if e - errors[-1] < 1e-3)     # first step within 0.001 of the final error
right.axvline(done, color=AXIS, linewidth=1, linestyle=":")
right.text(done + 70, 40, f"flat after ~{done:,} steps", color=INK2, fontsize=9)
right.annotate(f"final error {errors[-1]:.2f}", xy=(5000, errors[-1]), xytext=(2600, errors[-1] * 6),
               color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=AXIS, shrinkB=4))

fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("plot_regression.png", dpi=160)
print("wrote plot_regression.png")


# =====================================================================================
# Figure 2: the iris models from train_sklearn.py, drawn on the two petal measurements
# =====================================================================================
iris = load_iris()
X, y = iris.data[:, 2:], iris.target                        # petal length, petal width (cm)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
colors = [BLUE, ORANGE, AQUA]
cmap = ListedColormap(colors)

models = {
    "Decision tree": DecisionTreeClassifier(random_state=0),
    "Logistic regression": LogisticRegression(max_iter=1000),
    "K nearest neighbors, k = 1": KNeighborsClassifier(n_neighbors=1),
    "K nearest neighbors, k = 5": KNeighborsClassifier(n_neighbors=5),
}

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), sharex=True, sharey=True)
fig.suptitle("train_sklearn.py's three models, plus k = 1 for comparison: four different borders between species",
             color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left")
axes = axes.ravel()

# Top-left: just the data, one colour per species.
ax = axes[0]
for k, name in enumerate(iris.target_names):
    ax.scatter(*X[y == k].T, s=48, color=colors[k], edgecolor=SURFACE, linewidth=1.2, zorder=3, label=name)
    cx, cy = X[y == k].mean(axis=0)
    ax.text(cx, cy + 0.45, name, color=INK2, fontsize=9, ha="center", fontweight="bold",
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5))
ax.set_title("The data: 150 flowers, 3 species", loc="left")
ax.legend(loc="upper left", title="Species", title_fontsize=9)

# The next four: each model's map of "which species lives where".
for ax, (name, model) in zip(axes[1:5], models.items()):
    model.fit(X_train, y_train)
    # Ask the model about every point on a fine grid, then tint the plot by its answer.
    xx, yy = np.meshgrid(np.linspace(0.5, 7.3, 300), np.linspace(-0.2, 2.8, 300))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors, alpha=0.22, zorder=1)
    ax.scatter(*X_train.T, s=48, c=y_train, cmap=cmap, vmin=0, vmax=2, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    wrong = model.predict(X_test) != y_test
    ax.scatter(*X_test[wrong].T, s=48, c=y_test[wrong], cmap=cmap, vmin=0, vmax=2, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.scatter(*X_test[wrong].T, s=150, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=5)
    ax.set_title(f"{name}: {model.score(X_test, y_test):.0%} on unseen flowers", loc="left")

for ax in axes[:5]:
    ax.set(xlim=(0.5, 7.3), ylim=(-0.2, 2.8))
    ax.grid(False)
for ax in (axes[2], axes[3], axes[4]):                      # panels with nothing below them
    ax.set_xlabel("Petal length (cm)")
    ax.tick_params(labelbottom=True)
for ax in (axes[0], axes[3]):
    ax.set_ylabel("Petal width (cm)")

# Bottom-right cell: the key to the model panels, instead of a sixth plot.
key = axes[5]
key.axis("off")
key.legend(handles=[Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=MUTED, markeredgecolor=SURFACE, label="dot = training flower (colour = true species)"),
                    Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=MUTED, markeredgecolor=INK, markeredgewidth=1.6,
                           label="ringed dot = a test flower the model got wrong\n(the test flowers it got right are not drawn)")],
           loc="upper left", bbox_to_anchor=(0.0, 0.92), title="How to read the model panels", title_fontsize=10, alignment="left")
key.text(0.0, 0.55, "Coloured zones: what the model would answer\nfor a flower at that spot.\n\n"
         "Drawn on petal length and width only so the\nborder fits on a flat page. train_sklearn.py\n"
         "uses all 4 measurements, which is why its\nprinted scores differ slightly.",
         transform=key.transAxes, color=INK2, fontsize=9, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("plot_iris.png", dpi=160)
print("wrote plot_iris.png")


# =====================================================================================
# Figure 3: the same idea, made for a client. Only the three shortlisted models,
# no file names or code in the titles, and the caveat spelled out on the chart.
# =====================================================================================
shortlist = {
    "Logistic regression": models["Logistic regression"],
    "K nearest neighbors": models["K nearest neighbors, k = 5"],
    "Decision tree": models["Decision tree"],
}
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), sharex=True, sharey=True)
fig.suptitle("Where each model draws the border between the three iris species", color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left")
for ax, (name, model) in zip(axes, shortlist.items()):
    xx, yy = np.meshgrid(np.linspace(0.5, 7.3, 300), np.linspace(-0.2, 2.8, 300))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors, alpha=0.22, zorder=1)
    ax.scatter(*X_train.T, s=48, c=y_train, cmap=cmap, vmin=0, vmax=2, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    wrong = model.predict(X_test) != y_test
    ax.scatter(*X_test[wrong].T, s=48, c=y_test[wrong], cmap=cmap, vmin=0, vmax=2, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.scatter(*X_test[wrong].T, s=150, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=5)
    ax.set_title(f"{name}: {model.score(X_test, y_test):.0%} of unseen flowers correct", loc="left")
    ax.set(xlim=(0.5, 7.3), ylim=(-0.2, 2.8), xlabel="Petal length (cm)")
    ax.grid(False)
axes[0].set_ylabel("Petal width (cm)")
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=colors[k], markeredgecolor=SURFACE, label=n)
                        for k, n in enumerate(iris.target_names)], loc="upper left", title="Species", title_fontsize=9)
fig.text(0.02, 0.012, "Coloured zones: the model's answer for a flower at that spot.  Dots: the flowers it learned from.  "
         "Ringed dots: test flowers it got wrong.\nDrawn on two of the four measurements so it fits on a page, "
         "so percentages differ slightly from the full four-measurement results.", color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.075, 1, 0.93))
fig.savefig("plot_for_client.png", dpi=160)
print("wrote plot_for_client.png")
