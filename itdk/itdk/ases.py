import os
import re
import threading
import concurrent.futures

import numpy as np
import pandas as pd
# from numba import jit
from progress.counter import Counter

from itdk.logger import create_logger


def get_all_AS(file_path):
    ASes = {}
    counter = Counter("AS Processed Lines ")
    with open(file_path, "r") as f:
        for line in f:
            if line[0] != "#":
                splited_line = re.split(r"\s", line)
                node_id = splited_line[1]
                as_name = splited_line[2]
                ASes[node_id] = as_name
            counter.next()
    return ASes


def save_ASes(ASes, file_path):
    pd.DataFrame(set(ASes.values()), columns=["ases"]).to_hdf(
        file_path, "ases", mode="a", min_itemsize={"ases": 10}
    )


# @jit(nopython=True)
def noised_location(X, inverse_idx, counts):
    k = 0
    noised_X = np.zeros_like(X)
    noise = np.random.normal(scale=1.5, size=(counts[counts != 1].sum(), 2))
    for i, l in enumerate(inverse_idx):
        x = X[i].copy()
        if counts[l] == 1:
            noised_X[i] = x
        else:
            k += 1
            counts[l] -= 1
            noised_X[i] = x + x * noise[k]
    return noised_X


def get_unique_index(X, add_noise=False):
    n_combination = 1
    if add_noise:
        _, index, inverse_idx, counts = np.unique(
            X, axis=0, return_index=True, return_inverse=True, return_counts=True)
        X = noised_location(X, inverse_idx, counts)
    _, index, inverse_idx = np.unique(
        X, axis=0, return_index=True, return_inverse=True)
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


def cartesian(nodes, n=None, out=None):
    dtype = nodes[0].dtype

    if n is None:
        n = np.prod([x.size for x in nodes])
    if out is None:
        out = np.zeros([n, len(nodes)], dtype=dtype)

    m = n // nodes[0].size
    out[:, 0] = np.repeat(nodes[0], m)
    if nodes[1:]:
        cartesian(nodes[1:], out=out[0:m, 1:])
        for j in range(1, nodes[0].size):
            out[j * m: (j + 1) * m, 1:] = out[0:m, 1:]
    return out


def create_indices(counts, n_topology, max_attempt=10, seed=12345):
    attempt = 0
    rng = np.random.default_rng(seed)
    max_bounds = np.repeat(counts.reshape(1, -1), n_topology, axis=0)
    topology_indices = rng.integers(
        np.zeros(len(counts)), max_bounds, dtype=np.uint32)
    unique_topology_indices = np.unique(topology_indices, axis=0)
    n_unique_topology = len(unique_topology_indices)
    while n_unique_topology < n_topology:
        if attempt > max_attempt:
            break
        max_bounds = np.repeat(
            counts.reshape(1, -1), n_topology - n_unique_topology, axis=0
        )
        topology_indices = rng.integers(
            np.zeros(len(counts)), max_bounds, dtype=np.uint32
        )
        topology_indices = np.concatenate(
            [unique_topology_indices, topology_indices], axis=0
        )
        unique_topology_indices = np.unique(topology_indices, axis=0)
        n_unique_topology = len(unique_topology_indices)
        attempt += 1
    return unique_topology_indices


def get_node_id(node_locations, labels, inverse_index):
    nodes = []
    for label in labels:
        mask = inverse_index == label
        node_indices = node_locations.iloc[mask, 0].values.astype("<U10")
        nodes.append(node_indices)
    return nodes


def get_text_topology_from_arr(topology_arr):
    text_topology = ""
    for top in topology_arr:
        for n in top[:-1]:
            text_topology += "{},".format(n)
        text_topology += "{}\n".format(top[-1])
    return text_topology


def get_text_topology_from_indices(topology_indices, nodes):
    text_topology = ""
    for top_idx in topology_indices:
        for i, idx in enumerate(top_idx[:-1]):
            text_topology += "{},".format(nodes[i][idx])
        text_topology += "{}\n".format(nodes[len(top_idx) - 1][top_idx[-1]])
    return text_topology


def save_info(as_name, n_nodes, n_unique_nodes):
    with open(os.path.join("data", "unique_nodes.csv"), "a") as f:
        f.write("{},{},{}\n".format(as_name, n_nodes, n_unique_nodes))


class Extractor:
    def __init__(self):
        self._lock = threading.Lock()

    def get_text_topology(self, as_name, node_locations, file_logger, base=1000000):
        X = node_locations.loc[:, ("latitude", "longitude")].values
        inverse_index, labels, counts, n_combination = get_unique_index(X)
        with self._lock:
            save_info(as_name, len(inverse_index), len(labels))
        nodes = get_node_id(node_locations, labels, inverse_index)
        if n_combination != 0 and n_combination <= 2 * base:
            topology_arr = cartesian(nodes, n_combination)
            text_topology = get_text_topology_from_arr(topology_arr)
            with self._lock:
                file_logger.info(
                    "AS {} done, using cartesian with size {}".format(
                        as_name, n_combination
                    )
                )
        else:
            n_topology = base
            topology_indices = create_indices(counts, n_topology)
            text_topology = get_text_topology_from_indices(
                topology_indices, nodes)
            with self._lock:
                file_logger.info(
                    "AS {} done, using random selection with size {}"
                    " for {} possible combinations".format(
                        as_name, topology_indices.shape[0], n_combination
                    )
                )
        return text_topology

    def extract(self, as_name, root, node_locations, file_logger):
        text_topology = self.get_text_topology(
            as_name, node_locations, file_logger)
        with open(os.path.join(root, "{}.csv".format(as_name)), "w") as f:
            f.write(text_topology)


def extract_topo_from_unique_positions(geo_ases_path):
    num_of_empty_ases = 0
    file_logger = create_logger("ases.log")
    root = os.path.join("data", "unique_nodes_per_ases")
    os.makedirs(root, exist_ok=True)
    extractor = Extractor()
    ases = pd.read_hdf(geo_ases_path, "ases").values.squeeze()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for as_name in ases:
            node_locations = pd.read_hdf(
                geo_ases_path,
                "geo",
                columns=["id", "latitude", "longitude"],
                where=["ases=='{}'".format(as_name)],
            )
            if node_locations.shape[0] == 0:
                num_of_empty_ases += 1
                continue
            executor.submit(
                extractor.extract, as_name, root, node_locations, file_logger
            )
    file_logger.info(
        "{} ASes do not have any node with geolocation position".format(
            num_of_empty_ases
        )
    )
