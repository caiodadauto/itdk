import os
import re

import threading
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from itdk.logger import create_logger


def is_empty(root_dir):
    files = [
        name for name in os.listdir(root_dir) if name.split(".")[-1] == "csv"
    ]
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


def get_AS(idx, nodes, node_ases):
    node_id = nodes[idx]
    try:
        AS = node_ases[node_id]
    except KeyError:
        AS = None
    return AS


def add_to_links(n1, n2, i1, i2, links, count_links, AS_name):
    if AS_name not in links:
        links[AS_name] = ""
        count_links[AS_name] = 0
    links[AS_name] += "{},{},{},{}\n".format(n1, n2, i1, i2)
    count_links[AS_name] += 1


def merge_dict(from_d, to_d):
    for k, v in from_d.items():
        if k in to_d:
            to_d[k] += v
        else:
            to_d[k] = v


class LinkParser:
    def __init__(self, node_ases, data_dir):
        self.links = {}
        self.intra_AS = 0
        self.inter_AS = 0
        self.without_AS = 0
        self.count_links = {}
        self.node_ases = node_ases
        self.lock = threading.Lock()
        self.data_dir = data_dir

    def parse_link(self, inter_nodes):
        nodes, interfaces = inter_nodes
        n_nodes = len(nodes)
        for i in range(n_nodes):
            AS_i = get_AS(i, nodes, self.node_ases)
            if AS_i is None:
                with self.lock:
                    self.without_AS += 1
                continue
            for j in range((i + 1), n_nodes):
                AS_j = get_AS(j, nodes, self.node_ases)
                if AS_j is None:
                    with self.lock:
                        self.without_AS += 1
                    continue
                if AS_i == AS_j:
                    with self.lock:
                        add_to_links(
                            nodes[i],
                            nodes[j],
                            interfaces[i],
                            interfaces[j],
                            self.links,
                            self.count_links,
                            AS_i,
                        )
                        self.intra_AS += 1
                else:
                    with self.lock:
                        add_to_links(
                            nodes[i],
                            nodes[j],
                            interfaces[i],
                            interfaces[j],
                            self.links,
                            self.count_links,
                            "edge_links",
                        )
                        self.inter_AS += 1

    def write_file(self, AS, text):
        with open(os.paht.join(self.data_dir, "{}.csv".format(AS)), "w") as f:
            f.write(text)


def save_links_for_ases(node_ases, inter_node_links, file_logger, data_dir):
    parser = LinkParser(node_ases, data_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for inter_nodes in inter_node_links:
            executor.submit(parser.parse_link, inter_nodes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for AS, text in parser.links.items():
            executor.submit(parser.write_file, AS, text)
    msg = (
        "Links were saved, where: {} were ignored,".format(parser.without_AS)
        + " {} were inter AS".format(parser.inter_AS)
        + " {} were intra AS".format(parser.intra_AS)
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
                if len(inter_node_links) == 10000:
                    partial_count = save_links_for_ases(
                        node_ases, inter_node_links, file_logger
                    )
                    merge_dict(partial_count, count_links)
                    inter_node_links = []
            counter.update()
    partial_count = save_links_for_ases(node_ases, inter_node_links, file_logger, data_dir)
    merge_dict(partial_count, count_links)
    file_logger.info("The extraction is done.")
    df = pd.DataFrame(
        list(zip(count_links.keys(), count_links.values())),
        columns=["AS", "Nlinks"],
    )
    df.to_csv(os.path.join("data", "link_count.csv"))
