import os

import numpy as np
import pandas as pd
import graph_tool as gt
from tqdm import tqdm
from numba import jit, prange, float64, int64

from itdk.logger import create_logger
from itdk.utils import get_unique_index


# class Interfaces:
#     def __init__(self, path):
#         with open(path, "r") as f:
#             drop = 0
#             self._interfaces = f.readlines()
#             for l in self._interfaces:
#                 if l[0] == "#":
#                     drop += 1
#             self._interfaces = self._interfaces[drop:]
#         self._sep = re.compile(r"\s+")
#
#     def get_node_interfaces(self, node_id):
#         idx = int(node_id[1:]) - 1
#         assets = self._sep.split(self._interfaces[idx].rstrip())
#         print(assets)
#         read_node_id = assets[1][0:-1]
#         if read_node_id != node_id:
#             raise ValueError(
#                 "The requested ID {} is different of the read one {}".format(
#                     node_id, read_node_id
#                 )
#             )
#         node_interfaces = assets[2:]
#         print(node_interfaces)


# def fill_one_ip_gap(links, prefix, addrs, rng):
#     links = np.ascontiguousarray(links)
#     links = np.ascontiguousarray(links)
#     for i in range(len(links)):
#         ip1 = links[i][0]
#         ip2 = links[i][1]
#         # if ip1 == np.nan and ip2 == np.nan:
#         # create
#         # if ip2 is np.nan:


# def inspect_addrs(links, interfaces, file_logger, seed=12345):
#     ip1 = links["ip1"]
#     ip2 = links["ip2"]
#     addrs = pd.concat([ip1, ip2]).fillna("").sort_values()
#     prefix = (
#         pd.concat([ip1, ip2]).apply(lambda s: ".".join(s.split(".")[0:-1])).unique()
#     )
#     ip1_na = links["ip1"].isna()
#     ip2_na = links["ip2"].isna()
#     # without_addrs = ip1_na & ip2_na
#     one_addrs = ~(ip1_na & ip2_na)
#     if addrs.empty:
#         # TODO: implement the synthetic IP generator
#         file_logger.info("The AS {} does not have associated addresses")
#         return -1
#     rng = np.random.default_rng(seed)
#     fill_one_ip_gap(
#         links.loc[:, ("ip1", "ip2")].values,
#         one_addrs.values.flatten(),
#         prefix.values.flatten(),
#         rng,
#     )
#     # fill_two_ip_gaps(links, without_addrs, prefix, rng)


def check_coords(coordinates):
    if np.any((coordinates > 2) | (coordinates < 0)):
        raise ValueError(
            "Latitude or longitude are out"
            " of bound (latitude: [-90, 90] and longitude: [-180, 180])"
        )


def information_from_as(geo_path, links_path, as_name, as_issues, scale=True):
    as_links_path = os.path.join(links_path, as_name + ".csv")
    node_locations = pd.read_hdf(
        geo_path,
        "geo_with_ases",
        columns=["id", "latitude", "longitude"],
        where=["ases=='{}'".format(as_name)],
    )

    if node_locations.shape[0] == 0:
        as_issues["empty_ases"] += 1
        return None, None
    if scale:
        degree_locations = node_locations.loc[:, ("latitude", "longitude")].values
        node_locations.loc[:, "latitude"] = (degree_locations[:, 0] / 90) + 1
        node_locations.loc[:, "longitude"] = (degree_locations[:, 1] / 180) + 1
        check_coords(node_locations.loc[:, ("latitude", "longitude")])
    try:
        links_df = pd.read_csv(as_links_path, index_col=False, header=None, dtype=str)
        links_df.columns = ["n1", "n2", "ip1", "ip2"]
    except FileNotFoundError:
        as_issues["non_link_ases"] += 1
        return None, None
    return node_locations, links_df


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


def parse_interfaces(interfaces, edges):
    def to_binary(addrs):
        addrs = addrs.split(".")
        int_ip = np.array(list(map(lambda s: int(s), addrs)))
        binary_addrs = np.fliplr(
            ((int_ip.reshape(-1, 1) & (1 << np.arange(8))) > 0).astype(np.uint8)
        )
        return binary_addrs.flatten()

    dtype = np.dtype([("ip", np.uint8, (2, 32)), ("order", int, (2))])
    edge_interfaces = np.ones(interfaces.shape[0], dtype=dtype)
    for i, (inters, edge) in enumerate(zip(interfaces, edges)):
        for j, (inter_str, node) in enumerate(zip(inters, edge)):
            if inter_str != "":
                inter_bin = to_binary(inter_str)
                edge_interfaces["ip"][i] = inter_bin
            edge_interfaces["order"][i][j] = node
    return edge_interfaces


@jit(float64[:](int64[:, :], float64[:, :]), parallel=True, fastmath=True)
def get_edge_weights(edges, locations):
    size = edges.shape[0]
    weights = np.zeros(edges.shape[0])
    weights = np.ascontiguousarray(weights)
    for i in prange(size):
        p, s = edges[i]
        weights[i] = np.linalg.norm(locations[p] - locations[s])
    return weights


def create_graph(locations, links, graph_path):
    # link_ip1 = links.loc[:, ("label1", "label2", "ip1")]
    # link_ip2 = links.loc[:, ("label2", "label1", "ip2")]
    # link_ip1.columns = ["label1", "label2", "ip"]
    # link_ip2.columns = ["label1", "label2", "ip"]

    # links = pd.concat([link_ip1, link_ip2], axis=0).sort_values(["label1"])
    # edges = links.loc[:, ("label1", "label2")].values
    # edge_interfaces = parse_interfaces(links.loc[:, "ip"].fillna("").values)
    # edge_weights = get_edge_weights(edges, locations)

    links = links.sort_values(["label1"])
    edges = links.loc[:, ("label1", "label2")].values
    edge_interfaces = parse_interfaces(
        links.loc[:, ("ip1", "ip2")].fillna("").values, edges
    )
    edge_weights = get_edge_weights(edges, locations)

    g = gt.Graph(directed=False)
    g.add_vertex(locations.shape[0])
    g.add_edge_list(edges)
    g.vp.pos = g.new_vp("vector<float>", vals=locations)
    g.ep.weight = g.new_ep("float", vals=edge_weights)
    g.ep.ip = g.new_ep("object", vals=edge_interfaces)
    # print(g.ep.ip.get_2d_array(list(range(32))).T)
    # print(next(iter(g.ep.ip))) Object
    g.save(graph_path)
    return g


def parse_locations(node_locations, minimum_nodes, as_issues, file_logger):
    X = node_locations.loc[:, ("latitude", "longitude")].values
    noised_X, labels, nodes, _ = get_unique_index(X, add_noise=True)
    node_locations.loc[:, ("latitude", "longitude")] = noised_X
    node_locations["labels"] = labels
    check_coords(noised_X)
    if nodes.shape[0] < minimum_nodes:
        as_issues["small_ases"] += 1
        return None

    prune_labels = node_locations.duplicated(subset=["labels"])
    node_locations = node_locations.loc[~prune_labels]
    node_locations = node_locations.sort_values(by=["labels"])
    if (prune_labels).sum() > 0:
        file_logger.info(
            "{} duplicated nodes were removed even after noise addition".format(
                (prune_labels).sum()
            )
        )
    return node_locations


def extract_graphs(
    geo_path,
    links_path,
    data_dir,
    file_logger,
    minimum_nodes=15,
):
    ases = pd.read_hdf(geo_path, "ases").values.squeeze()
    as_issues = dict(empty_ases=0, small_ases=0, non_link_ases=0)
    for as_name in tqdm(ases):
        node_locations, links = information_from_as(
            geo_path, links_path, as_name, as_issues
        )
        if node_locations is None or links is None:
            continue
        file_logger.info(
            "AS: {}, raw nodes: {}, raw links: {}".format(
                as_name,
                node_locations.shape[0],
                links.shape[0],
            )
        )
        tmp = parse_locations(node_locations, minimum_nodes, as_issues, file_logger)
        if tmp is None:
            print("Skiped: Graph size < {}".format(minimum_nodes))
            continue
        node_locations = tmp
        locations = node_locations.loc[:, ("latitude", "longitude")].values
        links_non_loops_multi = parse_links(node_locations, links)
        graph_path = os.path.join(data_dir, "{}.gt.xz".format(as_name))
        g = create_graph(locations, links_non_loops_multi, graph_path)

        n_nodes = g.num_vertices()
        n_links = links_non_loops_multi.shape[0]
        file_logger.info(
            "AS: {}, nodes: {}, links: {}".format(as_name, n_nodes, n_links)
        )
    file_logger.info(
        "{} ASes do not present at least {} distinguish geolocated nodes;"
        "{} ASes do not have geolocated nodes; "
        "{} ASes do not have links; ".format(
            as_issues["small_ases"],
            minimum_nodes,
            as_issues["empty_ases"],
            as_issues["non_link_ases"],
        )
    )


def create_graphs_from_ases(geo_path, links_path):
    log_dir = "logs"
    data_dir = os.path.join("data", "raw_graphs_lack_inters")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "graphs.log")
    file_logger = create_logger(log_path)
    extract_graphs(geo_path, links_path, data_dir, file_logger)
