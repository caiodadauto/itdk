import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import seaborn as sns
import graph_tool as gt
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.utils import check_random_state
from sklearn.manifold import spectral_embedding
from sklearn.utils import as_float_array
from graph_tool.spectral import adjacency
from graph_tool.topology import label_components


N_GRAPHS = 0


def discretize(
    vectors, *, copy=True, max_svd_restarts=30, n_iter_max=20, random_state=None
):
    from scipy.sparse import csc_matrix
    from scipy.linalg import LinAlgError

    random_state = check_random_state(random_state)

    vectors = as_float_array(vectors, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    # Normalize the eigenvectors to an equal length of a vector of ones.
    # Reorient the eigenvectors to point in the negative direction with respect
    # to the first element.  This may have to do with constraining the
    # eigenvectors to lie in a specific quadrant to make the discretization
    # search easier.
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # Normalize the rows of the eigenvectors.  Samples should lie on the unit
    # hypersphere centered at the origin.  This transforms the samples in the
    # embedding space to the space of partition matrices.
    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False

    # If there is an exception we try to randomize and rerun SVD again
    # do this max_svd_restarts times.
    while (svd_restarts < max_svd_restarts) and not has_converged:

        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components),
            )

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError("SVD did not converge")
    return labels


def spectral_clustering(
    affinity,
    n_clusters=2,
    n_components=None,
    eigen_solver=None,
    random_state=None,
    n_init=10,
    eigen_tol=0.0,
    assign_labels="kmeans",
    norm_laplacian=False,
    drop_first=False,
):
    if assign_labels not in ("kmeans", "discretize"):
        raise ValueError(
            "The 'assign_labels' parameter should be "
            "'kmeans' or 'discretize', but '%s' was given" % assign_labels
        )

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components
    maps = spectral_embedding(
        affinity,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        norm_laplacian=norm_laplacian,
        eigen_tol=eigen_tol,
        drop_first=drop_first,
    )

    if assign_labels == "kmeans":
        _, labels, _ = k_means(
            maps, n_clusters, random_state=random_state, n_init=n_init
        )
    else:
        labels = discretize(maps, random_state=random_state)
    return labels


def check_components(g, min_graph_size, max_graph_size, save_dir):
    global N_GRAPHS
    components = dict(valid=[], to_cluster=[])
    component_labels, counts = label_components(g)
    labels = np.array(component_labels.a)
    unique_labels = np.arange(counts.shape[0])
    if len(counts) == 1:
        size = g.num_vertices()
        if size >= min_graph_size and size <= max_graph_size:
            components["valid"].append(g)
            g.save(os.path.join(save_dir, "{}.gt.xz".format(N_GRAPHS)))
            N_GRAPHS += 1
        elif size > max_graph_size:
            components["to_cluster"].append(g)
        elif size < min_graph_size:
            print("Invalid component with size: {}".format(size))
        return components

    valid_mask = (counts >= min_graph_size) & (counts <= max_graph_size)
    invalid_mask = counts < min_graph_size
    to_cluster_mask = counts > max_graph_size
    valid_labels = unique_labels[valid_mask]
    to_cluster_labels = unique_labels[to_cluster_mask]
    for l in valid_labels:
        gv = gt.GraphView(g, vfilt=labels == l)
        gv.save(os.path.join(save_dir, "{}.gt.xz".format(N_GRAPHS)))
        components["valid"].append(gv)
        N_GRAPHS += 1
    for l in to_cluster_labels:
        components["to_cluster"].append(gt.GraphView(g, vfilt=labels == l))
    print(
        "Invalid components: {} Sizes {}".format(
            invalid_mask.sum(), counts[invalid_mask]
        )
    )
    return components


def check_clusters(
    g, labels, unique_labels, counts, min_graph_size, max_graph_size, save_dir
):
    global N_GRAPHS
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
        components = check_components(gv, min_graph_size, max_graph_size, save_dir)
        to_cluster_graphs += components["to_cluster"]
        for gv in components["valid"]:
            gv.save(os.path.join(save_dir, "{}.gt.xz".format(N_GRAPHS)))
            N_GRAPHS += 1
    for l in to_cluster_labels:
        gv = gt.GraphView(g, vfilt=labels == l)
        to_cluster_graphs.append(gv)
    for l in invalid_labels:
        nodes_to_remove[labels == l] = -1
    gv = gt.GraphView(g, vfilt=nodes_to_remove == -1)
    components = check_components(gv, min_graph_size, max_graph_size, save_dir)
    to_cluster_graphs += components["to_cluster"]
    return to_cluster_graphs


def _get_clusters(
    gv,
    min_graph_size,
    max_graph_size,
    drop_first=True,
    norm_laplacian=False,
    save_dir=".",
):
    graph_size = gv.num_vertices()
    A = adjacency(gv, weight=gv.ep.weight)

    eigen_solver = None
    if graph_size < 100:
        n_components = 2
    elif graph_size >= 100 and graph_size < 1000:
        n_components = 4
    elif graph_size >= 1000 and graph_size < 100000:
        n_components = 8
    else:
        n_components = 10
        eigen_solver = "amg"

    labels = spectral_clustering(
        A,
        n_clusters=n_components,
        eigen_solver=eigen_solver,
        assign_labels="discretize",
        drop_first=drop_first,
        norm_laplacian=norm_laplacian,
    )
    unique_labels, counts = np.unique(labels, return_counts=True)
    to_cluster_graphs = check_clusters(
        gv, labels, unique_labels, counts, min_graph_size, max_graph_size, save_dir
    )
    # valid_graphs = components["valid"]
    return to_cluster_graphs


def get_clusters(
    to_cluster_graphs,
    min_graph_size,
    max_graph_size,
    n_threads,
    drop_first=True,
    norm_laplacian=False,
    save_dir=".",
    parallel=False,
):
    chunksize = 1
    partial_func = partial(
        _get_clusters,
        min_graph_size=min_graph_size,
        max_graph_size=max_graph_size,
        save_dir=save_dir,
    )
    if parallel:
        with Pool(n_threads) as pool:
            while len(to_cluster_graphs) > 0:
                next_to_cluster_graphs = []
                for out in pool.imap_unordered(
                    partial_func, to_cluster_graphs, chunksize=chunksize
                ):
                    next_to_cluster_graphs += out
                to_cluster_graphs = next_to_cluster_graphs
                n_graphs_to_cluster = len(to_cluster_graphs)
                if n_graphs_to_cluster > 500 and n_graphs_to_cluster <= 1000:
                    chunksize = 10
                elif n_graphs_to_cluster > 1000 and n_graphs_to_cluster <= 10000:
                    chunksize = 100
                elif n_graphs_to_cluster > 10000:
                    chunksize = 1000
    else:
        while len(to_cluster_graphs) > 0:
            gv = to_cluster_graphs.pop()
            to_cluster_graphs = partial_func(gv) + to_cluster_graphs


# def draw(g, valid_graphs, save_dir):
#     from graph_tool.draw import graph_draw
#
#     graph_draw(g, output="original.png")
#     for i, gv in enumerate(valid_graphs):
#         graph_draw(gv, output=os.path.join(save_dir, "{}.png".format(i)))


def run_clustering(
    save_dir,
    graph_paths,
    min_graph_size,
    max_graph_size,
    n_threads_to_file,
    n_threads_to_cluster,
    parallel,
):
    def graph_slicer(graph_paths, n_threads):
        n_files = len(graph_paths)
        chunk_size = n_files // n_threads
        idxs = np.arange(0, n_files, chunk_size)
        idxs = np.concatenate([idxs, [n_files]])
        for i in idxs[1:]:
            return graph_paths[i - 1 : i]

    def run(paths, min_graph_size, max_graph_size, save_dir):
        for gp in paths:
            g = gt.load_graph(gp)
            if g.num_vertices() > 1000:
                parallel = True
            else:
                parallel = False
            components = check_components(g, min_graph_size, max_graph_size, save_dir)
            get_clusters(
                components["to_cluster"],
                min_graph_size,
                max_graph_size,
                n_threads_to_cluster,
                save_dir=save_dir,
                parallel=parallel,
            )

    partial_func = partial(
        run,
        save_dir=save_dir,
        min_graph_size=min_graph_size,
        max_graph_size=max_graph_size,
    )
    if parallel:
        slicer = graph_slicer(graph_paths, n_threads_to_file)
        with Pool(n_threads_to_file) as pool:
            for _ in pool.imap_unordered(partial_func, slicer):
                print("Slicer Done")
    else:
        partial_func(graph_paths)


if __name__ == "__main__":
    dir_path = sys.argv[1]
    min_graph_size = 20
    max_graph_size = 60
    n_threads_to_file = int(np.ceil(os.cpu_count() * 0.6))
    n_threads_to_cluster = os.cpu_count() - n_threads_to_file

    graph_paths = []
    for p in os.listdir(dir_path):
        sp = p.split(".")
        if len(sp) > 2 and sp[1] == "gt":
            graph_paths.append(p)
    if len(graph_paths) / n_threads_to_file > 1:
        parallel = True
    else:
        parallel = False

    save_dir = os.path.join(dir_path, "graphs_clusters")
    os.makedirs(save_dir, exist_ok=True)
    run_clustering(
        save_dir,
        graph_paths,
        min_graph_size,
        max_graph_size,
        n_threads_to_file,
        n_threads_to_cluster,
        parallel,
    )
