import os

import numpy as np
import pandas as pd
import graph_tool as gt
from numba import jit, float64, int64

from itdk.logger import create_logger
from itdk.utils import get_unique_index


def check_coords(coordinates):
    if np.any((coordinates > 2) | (coordinates < 0)):
        raise ValueError(
            "Latitude or longitude are out"
            " of bound (latitude: [-90, 90] and longitude: [-180, 180])"
        )


def information_from_as(geo_path, links_path, interfaces_path, as_name, scale=True):
    as_interfaces_path = os.path.join(interfaces_path, as_name + ".csv")
    as_links_path = os.path.join(links_path, as_name + ".csv")
    node_locations = pd.read_hdf(
        geo_path,
        "geo",
        columns=["id", "latitude", "longitude"],
        where=["ases=='{}'".format(as_name)],
    )
    if node_locations.shape[0] == 0:
        return -1
    if scale:
        degree_locations = node_locations.loc[:, ("latitude", "longitude")].values
        node_locations.loc[:, "latitude"] = (degree_locations[:, 0] / 90) + 1
        node_locations.loc[:, "longitude"] = (degree_locations[:, 1] / 180) + 1
        check_coords(node_locations.loc[:, ("latitude", "longitude")])
    try:
        links_df = pd.read_csv(as_links_path, index_col=False, header=None, dtype=str)
        links_df.columns = ["n1", "n2", "ip1", "ip2"]
    except FileNotFoundError:
        return -2
    try:
        interfaces_df = pd.read_csv(
            as_interfaces_path, index_col=False, header=None, dtype=str
        )
        interfaces_df.columns = ["addrs", "id"]
        interfaces_df = interfaces_df.set_index("id")
    except FileNotFoundError:
        return -3
    return node_locations, links_df, interfaces_df


def parse_links(nodes, links):
    def to_label(s):
        try:
            return nodes.loc[s, "labels"]
        except KeyError:
            return -1

    links = links.copy()
    nodes = nodes.set_index("id")
    link_labels = links.loc[:, ("n1", "n2")].applymap(to_label)
    links["label1"] = link_labels["n1"]
    links["label2"] = link_labels["n2"]
    links_with_labels = links.loc[(links["label1"] >= 0) & (links["label2"] >= 0)]
    links_non_loops = links_with_labels.loc[
        links_with_labels["label1"] != links_with_labels["label2"]
    ]
    links_non_loops_multi = links_non_loops.loc[
        ~links_non_loops.duplicated(subset=["label1", "label2"])
    ]
    return links_non_loops_multi


def parse_interfaces(nodes, interfaces, as_name, file_logger):
    def to_binary(addrs):
        binary_interfaces = np.zeros((4, 8, len(addrs)), dtype=np.int8)
        for i, str_ip in enumerate(addrs):
            str_ip = str_ip.split(".")
            int_ip = np.array(list(map(lambda s: int(s), str_ip)))
            binary_ip = np.fliplr(
                ((int_ip.reshape(-1, 1) & (1 << np.arange(8))) > 0).astype(np.int8)
            )
            binary_interfaces[:, :, i] = binary_ip
        return binary_interfaces

    node_ids = nodes["id"]
    interfaces_of_nodes = []
    for node_id in node_ids:
        try:
            node_addrs = interfaces.loc[node_id].values.flatten()
        except KeyError:
            file_logger.warning("Node {} in AS {} does not have any interface".format(node_id, as_name))
            node_addrs = []
        interfaces_of_nodes.append(to_binary(node_addrs))
    return interfaces_of_nodes


@jit(float64[:](int64[:, :], float64[:, :]))
def get_edge_weights(edges, locations):
    size = edges.shape[0]
    weights = np.zeros(edges.shape[0])
    for i in range(size):
        p, s = edges[i]
        weights[i] = np.linalg.norm(locations[p] - locations[s])
    return weights


def create_graphs(locations, interfaces, edges, edge_weights, graph_path):
    g = gt.Graph(directed=False)
    g.add_vertex(locations.shape[0])
    g.vp.pos = g.new_vp("vector<float>", vals=locations)
    g.vp.interface = g.new_vp("object", vals=interfaces)
    g.add_edge_list(edges)
    g.ep.weight = g.new_ep("float", vals=edge_weights)
    g.save(graph_path)


def extract_graphs(
    geo_path,
    links_path,
    interfaces_path,
    root_path,
    file_logger,
    minimum_nodes=20,
):
    empty_ases = 0
    small_ases = 0
    non_link_ases = 0
    non_interfaces_ases = 0
    ases = pd.read_hdf(geo_path, "ases").values.squeeze()
    for as_name in ases:
        inf_out = information_from_as(geo_path, links_path, interfaces_path, as_name)
        if type(inf_out) is int:
            if inf_out == -1:
                empty_ases += 1
                continue
            elif inf_out == -2:
                non_link_ases += 1
                continue
            else:
                non_interfaces_ases += 1
                continue  # TODO: generate synthetic interfaces optionally
        node_locations, links, interfaces = inf_out
        print("AS", as_name, "nodes", node_locations.shape[0], "links", links.shape[0], "interfaces", interfaces.shape[0])

        X = node_locations.loc[:, ("latitude", "longitude")].values
        noised_X, labels, nodes, _, _ = get_unique_index(X, add_noise=True)
        node_locations.loc[:, ("latitude", "longitude")] = noised_X
        node_locations["labels"] = labels
        check_coords(noised_X)
        if nodes.shape[0] < minimum_nodes:
            small_ases += 1
            continue

        prune_labels = node_locations.duplicated(subset=["labels"])
        node_locations = node_locations.loc[~prune_labels]
        node_locations = node_locations.sort_values(by=["labels"])
        links_non_loops_multi = parse_links(node_locations, links)

        edges = links_non_loops_multi.loc[:, ("label1", "label2")].values
        locations = node_locations.loc[:, ("latitude", "longitude")].values
        edge_weights = get_edge_weights(edges, locations)
        interfaces_of_nodes = parse_interfaces(node_locations, interfaces, as_name, file_logger)
        graph_path = os.path.join(root_path, "{}.gt.xz".format(as_name))
        create_graphs(locations, interfaces_of_nodes, edges, edge_weights, graph_path)

        n_nodes = nodes.shape[0]
        all_n_nodes = labels.shape[0]
        n_links = links_non_loops_multi.shape[0]
        all_n_links = links.shape[0]
        file_logger.info(
            "For AS {},  {} of {} nodes and {} of {} links.".format(
                as_name, n_nodes, all_n_nodes, n_links, all_n_links
            )
        )
    file_logger.info(
        "{} ASes do not present at least {} distinguish geolocated nodes;"
        "{} ASes do not have geolocated nodes; "
        "{} ASes do not have links; "
        "{} ASes do not have interfaces".format(
            small_ases,
            minimum_nodes,
            empty_ases,
            non_link_ases,
            non_interfaces_ases,
        )
    )


def create_graphs_from_ases(geo_path, links_path, interfaces_path):
    root_path = os.path.join("data", "graphs")
    os.makedirs(root_path, exist_ok=True)
    file_logger = create_logger("graphs.log")
    extract_graphs(geo_path, links_path, interfaces_path, root_path, file_logger)
