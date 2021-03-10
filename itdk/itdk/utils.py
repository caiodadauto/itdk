import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# @jit(nopython=True)
def noised_location(X, inverse_idx, counts, mean, std, bound, seed=12345):
    k = 0
    counts = counts.copy()
    noised_X = np.zeros_like(X)
    rng = np.random.default_rng(12345)
    noise = rng.normal(loc=mean, scale=std, size=(counts[counts != 1].sum(), 2))
    for i, l in enumerate(inverse_idx):
        x = X[i].copy()
        if counts[l] == 1:
            noised_X[i] = x
        else:
            k += 1
            counts[l] -= 1
            new_x = x + x * noise[k]
            delta = bound[1] - bound[0]
            up_cmp = new_x > bound[1]
            down_cmp = new_x < bound[0]
            if np.any(up_cmp):
                eps = new_x[up_cmp] - bound[1]
                new_x[up_cmp] -= eps + rng.uniform(size=up_cmp.sum()) * delta
            elif np.any(down_cmp):
                eps = bound[0] - new_x[down_cmp]
                new_x[down_cmp] += eps + rng.uniform(size=down_cmp.sum()) * delta
            noised_X[i] = new_x
    return noised_X


def get_unique_index(X, add_noise=False, mean=0, std=0.8, bound=[0, 2]):
    n_combination = 1
    if add_noise:
        _, index, inverse_idx, counts = np.unique(
            X, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        X = noised_location(X, inverse_idx, counts, mean, std, bound)
    _, index, inverse_idx = np.unique(X, axis=0, return_index=True, return_inverse=True)
    n_labels = len(index)
    counts = np.zeros(n_labels)
    labels = np.arange(0, n_labels, 1)
    for label in labels:
        c = (inverse_idx == label).sum()
        counts[label] = c
        n_combination *= c
    if add_noise:
        return X, inverse_idx, labels, counts, n_combination
    return inverse_idx, labels, counts, n_combination


def get_graph_samples(df_slice, key_col="Num of Nodes"):
    si = 0
    num_of_top = df_slice["Num of Topologies"].values
    keys = df_slice[key_col].values
    graph_samples = np.zeros(num_of_top.sum(), dtype=int)
    for i, nt in enumerate(num_of_top):
        ei = si + nt
        graph_samples[si:ei] = nt * [keys[i]]
        si = ei
    return graph_samples


def draw_interval_dist(df, interval, min, max, path):
    begin_num_of_nodes = min
    while begin_num_of_nodes + interval < max:
        df_slice = df.loc[
            (df["Num of Nodes"] >= begin_num_of_nodes)
            & (df["Num of Nodes"] < begin_num_of_nodes + 50)
        ]
        graph_samples = get_graph_samples(df_slice)
        ax = sns.catplot(
            y="Num of Nodes",
            kind="count",
            palette="pastel",
            edgecolor=".6",
            data=pd.DataFrame(graph_samples, columns=["Num of Nodes"]),
        )
        ax.set_xlabels("Number of Graphs")
        ax.ax.yaxis.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                path, "{}_{}.pdf".format(begin_num_of_nodes, begin_num_of_nodes + 50)
            ),
            dpi=200,
        )


def get_dist_nodes_in_intervals(
    path_n_nodes_ases, path_dir, interval=50, min=30, max=0
):
    os.makedirs(path_dir, exist_ok=True)
    df = pd.read_csv(path_n_nodes_ases, index_col=None, header=None)
    df.columns = ["AS Name", "Num of Topologies", "Num of Nodes"]
    if max <= 0:
        max = df["Num of Nodes"].max()
    draw_interval_dist(df, interval, min, max, path_dir)


def get_dist_nodes(path_n_nodes_ases, path_dir, min=30, sort=False):
    os.makedirs(path_dir, exist_ok=True)
    df = pd.read_csv(path_n_nodes_ases, index_col=None, header=None)
    df.columns = ["AS Name", "Num of Topologies", "Num of Nodes"]
    df = df.drop("AS Name", axis=1)
    df = df[df["Num of Nodes"] >= min]
    df_grouped = df.groupby(["Num of Nodes"]).sum()
    df_grouped["Num of Nodes"] = df_grouped.index
    if sort:
        df_grouped.sort_values(by=["Num of Topologies"], inplace=True, ascending=False)
        graph_samples = get_graph_samples(df_grouped)
        graph_samples = graph_samples.astype(str)
    else:
        graph_samples = get_graph_samples(df_grouped)
    ax = sns.catplot(
        x="Num of Nodes",
        kind="count",
        palette="pastel",
        edgecolor=".6",
        data=pd.DataFrame(graph_samples, columns=["Num of Nodes"]),
        height=6,
        aspect=3.5,
    )
    plt.xticks(rotation=90)
    plt.yscale("log")
    ax.set_ylabels("Number of Graphs")
    ax.ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "all_distribution.pdf"), dpi=250)


def get_ases_contributions(path_n_nodes_ases, path_dir, min=30):
    os.makedirs(path_dir, exist_ok=True)
    as_contributions_pdf = PdfPages(os.path.join(path_dir, "ases_contributions.pdf"))
    df = pd.read_csv(path_n_nodes_ases, index_col=None, header=None)
    df.columns = ["AS Name", "Num of Topologies", "Num of Nodes"]
    df = df[df["Num of Nodes"] >= min]

    as_names = df["AS Name"].unique()
    as_names.sort()
    graph_sizes = df["Num of Nodes"].unique()
    graph_sizes.sort()

    for num_of_nodes in graph_sizes:
        df_aux = df.loc[df["Num of Nodes"] == num_of_nodes]
        graph_samples = get_graph_samples(df_aux, key_col="AS Name")
        ax = sns.catplot(
            x="ASes",
            kind="count",
            palette="pastel",
            edgecolor=".6",
            data=pd.DataFrame(graph_samples, columns=["ASes"]),
            height=3,
            aspect=2,
        )
        plt.xticks(rotation=90)
        plt.yscale("log")
        ax.set_ylabels("Number of Graphs")
        ax.ax.yaxis.grid(True)
        plt.title("For graphs with size {}".format(num_of_nodes))
        plt.tight_layout()
        as_contributions_pdf.savefig(ax.fig, dpi=250)
        plt.close()
    as_contributions_pdf.close()
