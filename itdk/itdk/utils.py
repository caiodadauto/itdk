import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
