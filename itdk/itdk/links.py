import os
import re
import gzip

import pandas as pd
from progress.counter import Counter

from itdk.logger import create_logger


def is_empty(root_dir):
    files = [
        name for name in os.listdir(root_dir) if name.split(".")[-1] == "csv"
    ]
    return len(files) == 0


def get_nodes_from_link(line):
    nodes = []
    splited_line = re.split(r"\s+", line)
    for raw_node in splited_line[2:-1]:
        nodes.append(raw_node.split(":")[0])
    return nodes


def get_AS(idx, nodes, node_ases, file_logger):
    node_id = nodes[idx]
    try:
        AS = node_ases[node_id]
    except KeyError:
        file_logger.info("Node {} does not have AS.".format(node_id))
        AS = None
    return AS


def add_to_links(n1, n2, links, count_links, AS_name):
    if AS_name not in links:
        links[AS_name] = ""
        count_links[AS_name] = 0
    links[AS_name] += "{},{}\n".format(n1, n2)
    count_links[AS_name] += 1


def merge_dict(from_d, to_d):
    for k, v in from_d.items():
        if k in to_d:
            to_d[k] += v
        else:
            to_d[k] = v


def save_links_for_ases(node_ases, node_links, file_logger):
    links = {}
    intra_AS = 0
    inter_AS = 0
    without_AS = 0
    count_links = {}
    for nodes in node_links:
        n_nodes = len(nodes)
        for i in range(n_nodes):
            AS_i = get_AS(i, nodes, node_ases, file_logger)
            if AS_i is None:
                without_AS += 1
                continue
            for j in range((i + 1), n_nodes):
                AS_j = get_AS(j, nodes, node_ases, file_logger)
                if AS_j is None:
                    without_AS += 1
                    continue
                if AS_i == AS_j:
                    add_to_links(nodes[i], nodes[j], links, count_links, AS_i)
                    intra_AS += 1
                else:
                    add_to_links(
                        nodes[i], nodes[j], links, count_links, "edge_links"
                    )
                    inter_AS += 1
    for AS, text in links.items():
        with open("data/links/{}.csv".format(AS), "a") as f:
            f.write(text)
    msg = (
        "Links were saved, where: {} were ignored,".format(without_AS)
        + " {} were inter AS".format(inter_AS)
        + " {} were intra AS".format(intra_AS)
    )
    file_logger.info(msg)
    return count_links


def extract_links_for_ases(link_path, geo_ases_path):
    node_links = []
    count_links = {}
    root_dir = "data/links"
    counter = Counter("Processed Links ")
    file_logger = create_logger("link.log")
    os.makedirs(root_dir, exist_ok=True)
    if not is_empty(root_dir):
        msg = "The directory {} is not empty.".format(root_dir)
        file_logger.error(msg)
        raise IOError(msg)
    node_ases = pd.read_hdf(geo_ases_path, "pandas", columns=["id", "ases"])
    node_ases.set_index("id", inplace=True)
    node_ases = node_ases.to_dict()["ases"]
    file_logger.info(
        "Start the link extraction from {}".format(os.path.abspath(link_path))
    )
    with open(link_path, "r") as file:
        for line in file:
            if line[0] != "#":
                node_links.append(get_nodes_from_link(line))
                if len(node_links) == 10000:
                    partial_count = save_links_for_ases(
                        node_ases, node_links, file_logger
                    )
                    merge_dict(partial_count, count_links)
                    node_links = []
            counter.next()
    partial_count = save_links_for_ases(node_ases, node_links, file_logger)
    merge_dict(partial_count, count_links)
    file_logger.info("The extraction is done.")
    df = pd.DataFrame(
        list(zip(count_links.keys(), count_links.values())),
        columns=["AS", "Nlinks"],
    )
    df.to_csv("data/link_count.csv")


def get_neighborhood_from_link(line):
    neighborhood = []
    splited_line = re.split(r"\s", line)
    node = splited_line[3].split(":")[0]
    for raw_node in splited_line[4:-2]:
        neighborhood.append(raw_node.split(":")[0])
    return node, neighborhood


def save_links(links, root_dir):
    for node, neighborhood in links.items():
        with gzip.GzipFile(
            os.path.join(root_dir, node + ".txt.gz"), "ab"
        ) as file:
            content = ""
            for neighbor in neighborhood:
                content += neighbor + "\n"
            file.write(content.encode())


def save_links_to_files(link_path):
    links = {}
    root_dir = "data/links/"
    counter = Counter("Processed Links ")
    file_logger = create_logger("link.log")
    os.makedirs(root_dir, exist_ok=True)
    if not is_empty(root_dir):
        msg = "The directory {} is not empty.".format(root_dir)
        file_logger.error(msg)
        raise IOError(msg)
    file_logger.info(
        "Start the link extraction from {}".format(os.path.abspath(link_path))
    )
    with open(link_path, "r") as f:
        for line in f:
            if line[0] != "#":
                node, neighborhood = get_neighborhood_from_link(line)
                if node in links:
                    links[node] += neighborhood
                else:
                    links[node] = neighborhood
                if len(links) == 1000:
                    save_links(links, root_dir)
                    links = {}
                counter.next()
    if len(links) > 0:
        save_links(links, root_dir)
    file_logger.info(
        "The links was extracted and saved in {}".format(
            os.path.abspath(root_dir)
        )
    )
