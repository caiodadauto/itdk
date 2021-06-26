import os
import re
import threading
import concurrent.futures
from random import shuffle

import numpy as np
import pandas as pd
from tqdm import tqdm

from itdk.logger import create_logger


def is_empty(root_dir):
    files = [name for name in os.listdir(root_dir) if name.split(".")[-1] == "csv"]
    return len(files) == 0


def get_inter_nodes_from_link(line):
    nodes = []
    interfaces = []
    splited_line = re.split(r"\s+", line)
    for raw_node in splited_line[2:-1]:
        nodes.append(raw_node.split(":")[0])
        if len(raw_node.split(":")) == 1:
            interfaces.append("")
        else:
            interfaces.append(raw_node.split(":")[1])
    return nodes, interfaces


def get_as(idx, nodes, node_ases):
    node_id = nodes[idx]
    try:
        as_name = node_ases[node_id]
    except KeyError:
        as_name = None
    return as_name


def add_to_links(n1, n2, i1, i2, links, count_links, as_name):
    if as_name not in links:
        links[as_name] = ""
        count_links[as_name] = 0
    links[as_name] += "{},{},{},{}\n".format(n1, n2, i1, i2)
    count_links[as_name] += 1


def merge_dict(from_d, to_d):
    for k, v in from_d.items():
        if k in to_d:
            to_d[k] += v
        else:
            to_d[k] = v


class LinkParser:
    def __init__(self, node_ases, data_dir):
        self.intra_as = 0
        self.inter_as = 0
        self.without_as = 0
        self.count_links = {}
        self.data_dir = data_dir
        self.node_ases = node_ases
        self.lock = threading.Lock()

    def parse_link(self, inter_nodes):
        links = {}
        count_links = {}
        inter_as = 0
        intra_as = 0
        without_as = 0
        nodes, interfaces = inter_nodes
        n_nodes = len(nodes)
        for i in range(n_nodes):
            as_i = get_as(i, nodes, self.node_ases)
            if as_i is None:
                without_as += 1
                continue
            for j in range((i + 1), n_nodes):
                as_j = get_as(j, nodes, self.node_ases)
                if as_j is None:
                    without_as += 1
                    continue
                if as_i == as_j:
                    add_to_links(
                        nodes[i],
                        nodes[j],
                        interfaces[i],
                        interfaces[j],
                        links,
                        count_links,
                        as_i,
                    )
                    intra_as += 1
                else:
                    add_to_links(
                        nodes[i],
                        nodes[j],
                        interfaces[i],
                        interfaces[j],
                        links,
                        count_links,
                        "edge_links",
                    )
                    inter_as += 1
        with self.lock:
            self.intra_as += intra_as
            self.inter_as += inter_as
            self.without_as += without_as
            merge_dict(count_links, self.count_links)
            for as_name, text in links.items():
                with open(os.path.join(self.data_dir, "{}.csv".format(as_name)), "a") as f:
                    f.write(text)


def save_links_for_ases(node_ases, inter_node_links, file_logger, data_dir):
    parser = LinkParser(node_ases, data_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for data in inter_node_links:
            executor.submit(parser.parse_link, data)
    msg = (
        "Links were saved, where: {} were ignored,".format(parser.without_as)
        + " {} were inter AS".format(parser.inter_as)
        + " {} were intra AS".format(parser.intra_as)
    )
    file_logger.info(msg)
    return parser.count_links


def extract_links_for_ases(link_path, geo_ases_path):
    count_links = {}
    inter_node_links = []
    log_dir = "logs"
    data_dir = os.path.join("data", "links")
    log_path = os.path.join(log_dir, "links.log")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    file_logger = create_logger(log_path)
    counter = tqdm("Processed lines [{}]".format(link_path), leave=False)
    if not is_empty(data_dir):
        msg = "The directory {} is not empty.".format(data_dir)
        file_logger.error(msg)
        raise IOError(msg)
    node_ases = pd.read_hdf(geo_ases_path, "geo_with_ases", columns=["id", "ases"])
    node_ases.set_index("id", inplace=True)
    node_ases = node_ases.to_dict()["ases"]
    with open(link_path, "r") as file:
        for line in file:
            if line[0] != "#":
                inter_node_links.append(get_inter_nodes_from_link(line))
                if len(inter_node_links) == 200000:
                    partial_count = save_links_for_ases(
                        node_ases, inter_node_links, file_logger, data_dir
                    )
                    merge_dict(partial_count, count_links)
                    inter_node_links = []
            counter.update()
    if len(inter_node_links) > 0:
        partial_count = save_links_for_ases(
            node_ases, inter_node_links, file_logger, data_dir
        )
        merge_dict(partial_count, count_links)
    file_logger.info("The extraction is done.")
    df = pd.DataFrame(
        list(zip(count_links.keys(), count_links.values())),
        columns=["AS", "Nlinks"],
    )
    df.to_csv(os.path.join("data", "link_count.csv"))
