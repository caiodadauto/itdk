import os

import numpy as np
import pandas as pd
import graph_tool as gt
from numba import jit
from graph_tool.draw import graph_draw

from itdk.logger import create_logger
from itdk.ases import get_unique_index


@jit(nopython=True)
def get_valid_nodes(n1, n2, n):
    n = np.sort(n)
    n1_idx = np.searchsorted(n, n1)
    n2_idx = np.searchsorted(n, n2)

    size_n = n.shape[0]
    ext_n = np.zeros(size_n + 1, dtype=np.int64)
    ext_n[0: n.shape[0]] = n

    n1_out = ext_n[n1_idx]
    n2_out = ext_n[n2_idx]
    n1_mask = n1_out == n1
    n2_mask = n2_out == n2
    mask = n1_mask & n2_mask

    senders = n1[mask]
    receivers = n2[mask]
    all_valid_nodes = np.concatenate((senders, receivers))
    return all_valid_nodes


def draw_graph(locations, edges, file_name, ext="png", use_pos=False):
    nodes_in_edges = edges.flatten()
    _, index, inverse_index = np.unique(
        nodes_in_edges, return_index=True, return_inverse=True)
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


def get_edges(n1, n2, n):
    all_valid_nodes = get_valid_nodes(n1, n2, n)
    unique_valid_nodes, reverse_index = np.unique(
        all_valid_nodes, return_inverse=True)
    return unique_valid_nodes, reverse_index


def extract_graphs_from_unique_nodes(unique_nodes_path, geo_path, links_path, assets_path, file_logger):
    as_names = [
        name for name in os.listdir(unique_nodes_path) if name.split(".")[-1] == "csv"
    ]
    for as_name_file in as_names:
        as_name = as_name_file.split(".")[0]
        node_locations = pd.read_hdf(
            geo_path,
            "geo",
            columns=["id", "latitude", "longitude"],
            where=["ases=='{}'".format(as_name)],
        )
        as_links_path = os.path.join(links_path, as_name_file)
        as_nodes_path = os.path.join(unique_nodes_path, as_name_file)
        try:
            links_df = pd.read_csv(
                as_links_path, index_col=False, header=None, dtype=str
            )
            links_df.columns = ["n1", "n2", "ip1", "ip2"]
            node_ids = links_df[["n1", "n2"]].applymap(lambda s: s[1:])
            nodes_1 = node_ids["n1"].values.squeeze().astype(int)
            nodes_2 = node_ids["n2"].values.squeeze().astype(int)
        except FileNotFoundError:
            file_logger.warning(
                "AS {} does not have any link.".format(
                    as_name_file.split(".")[0])
            )
            continue
        with open(as_nodes_path, "r") as f:
            k = 0
            for node_group in f:
                node_group = node_group.rstrip()
                node_group = node_group.split(",")
                node_group = np.array(
                    list(map(lambda s: int(s[1:]), node_group)))
                nodes, flat_edges = get_edges(nodes_1, nodes_2, node_group)

                if len(nodes) == 0:
                    print("Skip line {}".format(k))
                    continue
                print(len(nodes))

                query_node_locations = node_locations.loc[
                    node_locations["id"].isin(
                        ["N{}".format(node_id) for node_id in nodes]
                    )
                ]
                query_node_locations = query_node_locations.sort_values("id")
                locations = query_node_locations.loc[:,
                                                     ("latitude", "longitude")].values
                edges = np.stack(
                    (flat_edges[0: len(flat_edges) // 2], flat_edges[len(flat_edges) // 2:]))
                graph_path = os.path.join(
                    assets_path, "{}-{}".format(as_name, k))
                draw_graph(locations, edges, graph_path)
                k += 1


def extract_graphs_from_joining_nodes(geo_path, links_path, assets_path, file_logger, add_noise):
    num_of_empty_ases = 0
    num_of_non_link_ases = 0
    num_of_small_unique_nodes_ases = 0
    ases = pd.read_hdf(geo_path, "ases").values.squeeze()
    for as_name in ases:
        if as_name != '2906':
            continue

        as_links_path = os.path.join(links_path, as_name + ".csv")
        try:
            links_df = pd.read_csv(
                as_links_path, index_col=False, header=None, dtype=str
            )
            links_df.columns = ["n1", "n2", "ip1", "ip2"]
        except FileNotFoundError:
            num_of_non_link_ases += 1
            continue

        node_locations = pd.read_hdf(
            geo_path,
            "geo",
            columns=["id", "latitude", "longitude"],
            where=["ases=='{}'".format(as_name)],
        )
        if node_locations.shape[0] == 0:
            num_of_empty_ases += 1
            continue

        if add_noise:
            X = node_locations.loc[:, ("latitude", "longitude")].values
            noised_X, labels, nodes, _, _ = get_unique_index(X, add_noise)
            node_locations.loc[:, ("latitude", "longitude")] = noised_X
            print(X.shape, noised_X.shape)
        else:
            labels, nodes, _, _ = get_unique_index(
                node_locations.loc[:, ("latitude", "longitude")].values
            )
        if nodes.shape[0] < 20:
            num_of_small_unique_nodes_ases += 1
            continue

        node_locations["labels"] = labels
        node_locations = node_locations.set_index("id")
        link_labels = links_df.loc[:, ("n1", "n2")].applymap(
            lambda s: node_locations.loc[s, "labels"])
        links_df["label1"] = link_labels["n1"]
        links_df["label2"] = link_labels["n2"]
        links_df = links_df.set_index(["n1", "n2"])
        links_non_loops = links_df.loc[links_df["label1"]
                                       != links_df["label2"]]
        links_non_loops_multi = links_non_loops.loc[~links_non_loops.duplicated(
            subset=["label1", "label2"])]

        n_nodes = nodes.shape[0]
        all_n_nodes = labels.shape[0]
        n_links = links_non_loops_multi.shape[0]
        all_n_links = links_df.shape[0]
        file_logger.info("For AS {},  {} nodes was used of {}, and {} links was used of {}.".format(
            as_name, n_nodes, all_n_nodes, n_links, all_n_links))

        prune_labels = node_locations.duplicated(subset=["labels"])
        node_unique_locations = node_locations.loc[~prune_labels]
        locations = node_unique_locations.loc[:,
                                              ("latitude", "longitude")].values
        edges = links_non_loops_multi.loc[:, ("label1", "label2")].values
        graph_path = os.path.join(
            assets_path, "{}".format(as_name))
        draw_graph(locations, edges, graph_path, ext="pdf")

        file_logger.info(
            "{} ASes do not have any node with geolocation position."
            "{} ASes with number of nodes with different positions less than 20".format(
                num_of_small_unique_nodes_ases, num_of_empty_ases
            )
        )


def create_graphs_from_ases(geo_path, links_path, unique_nodes_path=None, add_noise=False):
    assets_path = os.path.join("data", "draws")
    os.makedirs(assets_path, exist_ok=True)
    file_logger = create_logger("graphs.log")
    if unique_nodes_path is not None:
        extract_graphs_from_unique_nodes(
            unique_nodes_path, geo_path, links_path, assets_path, file_logger
        )
    else:
        extract_graphs_from_joining_nodes(
            geo_path, links_path, assets_path, file_logger, add_noise)


create_graphs_from_ases(
    "/home/caio/Documents/university/Ph.D./topology/CAIDA/itdk/data/itdk.h5",
    "/home/caio/Documents/university/Ph.D./topology/CAIDA/itdk/data/links/",
    add_noise=True
    # "/home/caio/Documents/university/Ph.D./topology/CAIDA/itdk/data/unique_nodes_per_ases/",
)
