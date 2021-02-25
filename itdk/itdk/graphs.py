import os

import numpy as np
import pandas as pd
import graph_tool as gt
from graph_tool.draw import graph_draw

from itdk.logger import create_logger
from itdk.ases import get_unique_index


def draw_graph(locations, edges, file_name, ext="png", use_pos=False):
    nodes_in_edges = edges.flatten()
    _, index, inverse_index = np.unique(
        nodes_in_edges, return_index=True, return_inverse=True
    )
    edges = inverse_index.reshape(edges.shape)
    locations = locations[nodes_in_edges[index]]

    G = gt.Graph(directed=False)
    G.add_vertex(len(index))
    G.vp.pos = G.new_vp("vector<float>", vals=locations)
    G.add_edge_list(edges)
    if use_pos:
        graph_draw(G, pos=G.vp.pos, output=file_name + "." + ext)
    else:
        graph_draw(G, output=file_name + "." + ext)


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
        interfaces_df.columns = ["i", "n"]
    except FileNotFoundError:
        return -3
    return node_locations, links_df, interfaces_df


def get_edges(nodes, links, labels):
    nodes["labels"] = labels
    nodes = nodes.set_index("id")
    link_labels = links.loc[:, ("n1", "n2")].applymap(
        lambda s: nodes.loc[s, "labels"]
    )
    links["label1"] = link_labels["n1"]
    links["label2"] = link_labels["n2"]
    links = links.set_index(["n1", "n2"])
    links_non_loops = links.loc[links["label1"] != links["label2"]]
    links_non_loops_multi = links_non_loops.loc[
        ~links_non_loops.duplicated(subset=["label1", "label2"])
    ]
    return links_non_loops_multi


def get_interfaces(nodes, interfaces):
    def to_binary(addrs):
        binary_interfaces = np.zeros((len(addrs), 32), dtype=np.int8)
        for i, str_ip in enumerate(addrs):
            str_ip = str_ip.split(".")
            int_ip = np.array(list(map(lambda s: int(s), str_ip)))
            binary_ip = ((int_ip.reshape(-1, 1) & (1 << np.arange(8))) > 0).astype(np.int8)
            binary_interfaces[i] = binary_ip.flatten()
        return binary_interfaces

    node_ids = nodes["id"]
    interfaces_of_nodes = []
    for node_id in node_ids:
        node_addrs = interfaces.loc[interfaces["id"] == node_id, "addrs"].values
        interfaces_of_nodes.append(to_binary(node_addrs))
    return interfaces_of_nodes


def extract_graphs_from_joining_nodes(
    geo_path,
    links_path,
    interfaces_path,
    assets_path,
    file_logger,
    add_noise,
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

        if add_noise:
            X = node_locations.loc[:, ("latitude", "longitude")].values
            noised_X, labels, nodes, _, _ = get_unique_index(X, add_noise)
            node_locations.loc[:, ("latitude", "longitude")] = noised_X
        else:
            labels, nodes, _, _ = get_unique_index(
                node_locations.loc[:, ("latitude", "longitude")].values
            )
        if nodes.shape[0] < 20:
            small_ases += 1
            continue
        
        links_non_loops_multi = get_edges(node_locations, links, labels)
        interfaces_of_nodes = get_interfaces(node_locations, interfaces)

        n_nodes = nodes.shape[0]
        all_n_nodes = labels.shape[0]
        n_links = links_non_loops_multi.shape[0]
        all_n_links = links.shape[0]
        file_logger.info(
            "For AS {},  {} / {} of nodes and {} / {} of links.".format(
                as_name, n_nodes, all_n_nodes, n_links, all_n_links
            )
        )

        prune_labels = node_locations.duplicated(subset=["labels"])
        node_unique_locations = node_locations.loc[~prune_labels]
        locations = node_unique_locations.loc[:, ("latitude", "longitude")].values
        edges = links_non_loops_multi.loc[:, ("label1", "label2")].values
        graph_path = os.path.join(assets_path, "{}".format(as_name))
        draw_graph(locations, edges, graph_path, ext="pdf")

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
    assets_path = os.path.join("data", "draws")
    os.makedirs(assets_path, exist_ok=True)
    file_logger = create_logger("graphs.log")
    extract_graphs_from_joining_nodes(
        geo_path, links_path, interfaces_path, assets_path, file_logger, add_noise
    )


create_graphs_from_ases(
    "/home/caio/Documents/university/Ph.D./topology/CAIDA/itdk/data/itdk.h5",
    "/home/caio/Documents/university/Ph.D./topology/CAIDA/itdk/data/links/",
    add_noise=True
    # "/home/caio/Documents/university/Ph.D./topology/CAIDA/itdk/data/unique_nodes_per_ases/",
)
