import os
from functools import partial
from multiprocessing.dummy import Pool, current_process

import numpy as np
import graph_tool as gt
from tqdm import tqdm
from graph_tool.spectral import adjacency
from graph_tool.topology import label_components
from sklearn.cluster import spectral_clustering


def save_graph(g, path):
    pruned_g = gt.Graph(g, prune=True)
    pruned_g.save(path)


def check_components(g, min_graph_size, max_graph_size, name):
    to_cluster_graphs = []
    component_labels, counts = label_components(g)[0:2]
    labels = np.array(component_labels.a)
    unique_labels = np.arange(counts.shape[0])
    if len(counts) == 1:
        size = g.num_vertices()
        if size >= min_graph_size and size <= max_graph_size:
            save_graph(
                g,
                os.path.join(
                    name["dir"], "{}-{}.gt.xz".format(name["AS"], name["cluster"])
                ),
            )
            name["cluster"] += 1
        elif size > max_graph_size:
            to_cluster_graphs.append(g)
        return to_cluster_graphs, size

    valid_mask = (counts >= min_graph_size) & (counts <= max_graph_size)
    invalid_mask = counts < min_graph_size
    to_cluster_mask = counts > max_graph_size
    valid_labels = unique_labels[valid_mask]
    to_cluster_labels = unique_labels[to_cluster_mask]
    for l in valid_labels:
        gv = gt.GraphView(g, vfilt=labels == l)
        save_graph(
            gv,
            os.path.join(
                name["dir"], "{}-{}.gt.xz".format(name["AS"], name["cluster"])
            ),
        )
        name["cluster"] += 1
    for l in to_cluster_labels:
        gv = gt.GraphView(g, vfilt=labels == l)
        to_cluster_graphs.append(gv)
    return to_cluster_graphs, sum(counts[invalid_mask])


def check_clusters(
    g, labels, unique_labels, counts, min_graph_size, max_graph_size, name
):
    n_removed_nodes = 0
    to_cluster_graphs = []
    invalid_mask = counts < min_graph_size
    valid_mask = (counts >= min_graph_size) & (counts <= max_graph_size)
    to_cluster_mask = counts > max_graph_size

    nodes_to_remove = np.zeros_like(labels)
    valid_labels = unique_labels[valid_mask]
    invalid_labels = unique_labels[invalid_mask]
    to_cluster_labels = unique_labels[to_cluster_mask]

    for l in valid_labels:
        gv = gt.GraphView(g, vfilt=labels == l)
        _, _n_removed_nodes = check_components(gv, min_graph_size, max_graph_size, name)
        n_removed_nodes += _n_removed_nodes
    for l in to_cluster_labels:
        gv = gt.GraphView(g, vfilt=labels == l)
        _to_cluster_graphs, _n_removed_nodes = check_components(
            gv, min_graph_size, max_graph_size, name
        )
        to_cluster_graphs += _to_cluster_graphs
        n_removed_nodes += _n_removed_nodes
    for l in invalid_labels:
        nodes_to_remove[labels == l] = -1
    gv = gt.GraphView(g, vfilt=nodes_to_remove == -1)
    _to_cluster_graphs, _n_removed_nodes = check_components(
        gv, min_graph_size, max_graph_size, name
    )
    to_cluster_graphs += _to_cluster_graphs
    n_removed_nodes += _n_removed_nodes
    return to_cluster_graphs, n_removed_nodes


def _get_clusters(
    gv,
    min_graph_size,
    max_graph_size,
    name,
):
    graph_size = gv.num_vertices()
    eigen_solver = None
    if graph_size < 100:
        n_components = 3
    elif graph_size >= 100 and graph_size < 1000:
        n_components = 5
    elif graph_size >= 1000 and graph_size < 100000:
        n_components = 9
    else:
        n_components = 11
        eigen_solver = "amg"
    A = adjacency(gv, weight=gv.ep.weight)

    try:
        labels = spectral_clustering(
            A,
            n_clusters=n_components,
            eigen_solver=eigen_solver,
            assign_labels="discretize",
        )
    except Exception:
        return [], -1

    unique_labels, counts = np.unique(labels, return_counts=True)
    to_cluster_graphs, n_removed_nodes = check_clusters(
        gv, labels, unique_labels, counts, min_graph_size, max_graph_size, name
    )
    return to_cluster_graphs, n_removed_nodes


def get_clusters(
    to_cluster_graphs,
    min_graph_size,
    max_graph_size,
    name,
):
    n_removed_nodes = 0
    while len(to_cluster_graphs) > 0:
        gv = to_cluster_graphs.pop()
        _to_cluster_graphs, _n_removed_nodes = _get_clusters(
            gv, min_graph_size, max_graph_size, name
        )
        if _n_removed_nodes < 0:
            return -1
        to_cluster_graphs = _to_cluster_graphs + to_cluster_graphs
        n_removed_nodes += _n_removed_nodes
    return n_removed_nodes


def _run_clustering(paths, save_dir, min_graph_size, max_graph_size):
    status = {}
    bar = tqdm(total=len(paths), desc="{}".format(current_process()))
    for gp in paths:
        as_name = os.path.basename(gp).split(".")[0]
        bar.set_postfix(AS=as_name)
        name = dict(dir=save_dir, cluster=0, AS=as_name)
        g = gt.load_graph(gp)
        to_cluster_graphs, n_removed_nodes = check_components(
            g, min_graph_size, max_graph_size, name
        )
        _n_removed_nodes = get_clusters(
            to_cluster_graphs,
            min_graph_size,
            max_graph_size,
            name,
        )
        if _n_removed_nodes < 0:
            sremoved = -1
        else:
            sremoved = n_removed_nodes + _n_removed_nodes
        status[as_name] = (g.num_vertices(), sremoved)
        bar.update()
    bar.close()
    return status


def run_clustering(
    save_dir,
    graph_dir,
    data_dir,
    min_graph_size,
    max_graph_size,
    n_threads,
    parallel,
):
    def graph_slicer(graph_paths, n_threads):
        n_files = len(graph_paths)
        chunk_size = n_files // n_threads
        if chunk_size < 0:
            chunk_size = n_files
        idxs = np.arange(0, n_files, chunk_size)
        idxs = np.concatenate([idxs, [n_files]])
        for i in range(1, len(idxs)):
            yield graph_paths[idxs[i - 1] : idxs[i]]

    graph_paths = []
    for p in os.listdir(graph_dir):
        sp = p.split(".")
        if len(sp) > 2 and sp[1] == "gt":
            graph_paths.append(os.path.join(graph_dir, p))
    os.makedirs(save_dir, exist_ok=True)

    partial_func = partial(
        _run_clustering,
        save_dir=save_dir,
        min_graph_size=min_graph_size,
        max_graph_size=max_graph_size,
    )
    finfo = open(os.path.join(data_dir, "cluster_info.csv"), "w")
    finfo.write("AS, Num Nodes, Removed Nodes\n")
    if parallel:
        slicer = graph_slicer(graph_paths, n_threads)
        with Pool(n_threads) as pool:
            for status in pool.imap_unordered(partial_func, slicer):
                for as_name, (num_nodes, num_removed_nodes) in status.items():
                    finfo.write(
                        "{}, {}, {}\n".format(as_name, num_nodes, num_removed_nodes)
                    )
    else:
        status = partial_func(graph_paths)
        for as_name, (num_nodes, num_removed_nodes) in status.items():
            finfo.write("{}, {}, {}\n".format(as_name, num_nodes, num_removed_nodes))
    finfo.close()
