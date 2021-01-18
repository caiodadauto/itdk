import os
import re

import numpy as np
import pandas as pd
from progress.counter import Counter


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


def get_unique_index(node_locations):
    labels = {}
    max_count = 0
    n_combination = 1
    X = node_locations[["latitude", "longitude"]].values
    v, index, inverse_idx = np.unique(
        X, axis=0, return_index=True, return_inverse=True
    )
    for label in range(len(index)):
        c = (inverse_idx == label).sum()
        labels[label] = c
        if c > max_count:
            max_count = c
        n_combination *= c
    return inverse_idx, labels, max_count, n_combination


def get_nodes_per_location(node_locations):
    nodes = []
    inverse_index, labels, max_count, n_combination = get_unique_index(
        node_locations
    )
    for label, n_labels in labels.item():
        mask = inverse_index == label
        node_indices = node_locations.iloc[mask, 0].values.astype("<U10")
        if n_labels < max_count:
            extra_indices = np.random.choice(
                node_indices, max_count - n_labels
            )
            node_indices = np.concatenate(
                [node_indices, extra_indices], axis=0
            )
        nodes.append(node_indices)
    return np.array(nodes).T, n_combination


def save_unique_groups(nodes, n_comobination, root, as_name, ratio=.4):
    def has_equal_topology(groups):
        if len(groups) == 0:
            return False
        last_group = groups[-1]
        for g in groups[:-1]:
            for line in last_group:
                if np.any(np.all(g == line, axis=1)):
                    return True
        return False

    dir_name = os.path.join(root, as_name)
    os.makedirs(dir_name, exist_ok=True)
    groups = []
    rng = np.random.default_rng()
    n_groups = int(np.ceil(n_comobination * ratio))
    for _ in range(n_groups):
        while(has_equal_topology(groups)):
            g = rng.permutation(nodes, axis=0)
        groups.append(g)
    groups = np.concatenate(groups, axis=0)
    for topology in groups:
        with open(os.path.join(dir_name, "{}.csv"), 'w') as f:
            text = ""
            for node in topology:
                text += "{}\n".format(node)
            f.write(text)


def group_nodes_with_unique_locations(geo_ases_path):
    root = os.path.join("data", "unique_noddes_per_ases")
    os.makedirs(root, exist_ok=True)
    ases = pd.read_hdf(geo_ases_path, "ases").values.squeeze()
    for as_name in ases:
        node_locations = pd.read_hdf(
            geo_ases_path,
            "geo",
            columns=["id", "latitude", "longitude"],
            where=["ases=={}".format(as_name)],
        )
        nodes, n_combination = get_nodes_per_location(node_locations)
        save_unique_groups(nodes, n_combination, as_name)
