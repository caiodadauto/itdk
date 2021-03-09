import os

import numpy as np
import pandas as pd
import graph_tool as gt

from itdk.logger import create_logger
from itdk.ases import get_unique_index


def information_from_as(geo_path, links_path, interfaces_path, as_name):
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
    except FileNotFoundError:
        return -3
    return node_locations, links_df, interfaces_df


def parse_links(nodes, links):
    def to_label(s):
        try:
            return nodes.loc[s, "labels"]
        except KeyError:
            return -1

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


def parse_interfaces(nodes, interfaces):
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
        node_addrs = interfaces.loc[interfaces["id"] == node_id, "addrs"].values
        interfaces_of_nodes += to_binary(node_addrs)
    return interfaces_of_nodes


def create_graphs(locations, edges, interfaces, graph_path):
    G = gt.Graph(directed=False)
    G.add_vertex(locations.size)
    G.vp.pos = G.new_vp("vector<float>", vals=locations)
    G.vp.pos = G.new_vp("object", vals=interfaces)
    G.add_edge_list(edges)


def extract_graphs(
    geo_path,
    links_path,
    interfaces_path,
    root_path,
    file_logger,
    minimum_nodes=30,
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

        X = node_locations.loc[:, ("latitude", "longitude")].values
        noised_X, labels, nodes, _, _ = get_unique_index(X, add_noise=True)
        node_locations.loc[:, ("latitude", "longitude")] = noised_X
        node_locations["labels"] = labels
        if nodes.shape[0] < 20:
            small_ases += 1
            continue

        prune_labels = node_locations.duplicated(subset=["labels"])
        node_locations = node_locations.loc[~prune_labels]
        node_locations = node_locations.sort_values(by=["labels"])
        links_non_loops_multi = parse_links(node_locations, links)

        edges = links.loc[:, ("label1", "label2")].values
        locations = node_locations.loc[:, ("latitude", "longitude")].values
        interfaces_of_nodes = parse_interfaces(node_locations, interfaces)
        graph_path = os.path.join(root_path, "{}".format(as_name))
        create_graphs(locations, edges, interfaces_of_nodes, graph_path)

        n_nodes = nodes.shape[0]
        all_n_nodes = labels.shape[0]
        n_links = links_non_loops_multi.shape[0]
        all_n_links = links.shape[0]
        file_logger.info(
            "For AS {},  {} / {} of nodes and {} / {} of links.".format(
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


def create_graphs_from_ases(geo_path, links_path, interfaces_path, add_noise=True):
    root_path = os.path.join("data", "graphs")
    os.makedirs(root_path, exist_ok=True)
    file_logger = create_logger("graphs.log")
    extract_graphs(
        geo_path, links_path, interfaces_path, root_path, file_logger, add_noise
    )
