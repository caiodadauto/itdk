import os
import re
from random import shuffle
from functools import partial

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

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


def write_link_file(data, data_dir):
    as_name, text = data
    with open(os.path.join(data_dir, "{}.csv".format(as_name)), "a") as f:
        f.write(text)


def link_parser(inter_nodes, node_ases):
    links = {}
    n_links = {}
    n_inter_links = 0
    n_intra_links = 0
    n_non_as_links = 0
    nodes, interfaces = inter_nodes
    n_nodes = len(nodes)
    for i in range(n_nodes):
        as_i = get_as(i, nodes, node_ases)
        if as_i is None:
            n_non_as_links += 1
            continue
        for j in range((i + 1), n_nodes):
            as_j = get_as(j, nodes, node_ases)
            if as_j is None:
                n_non_as_links += 1
                continue
            if as_i == as_j:
                add_to_links(
                    nodes[i],
                    nodes[j],
                    interfaces[i],
                    interfaces[j],
                    links,
                    n_links,
                    as_i,
                )
                n_intra_links += 1
            else:
                add_to_links(
                    nodes[i],
                    nodes[j],
                    interfaces[i],
                    interfaces[j],
                    links,
                    n_links,
                    "edge_links",
                )
                n_inter_links += 1
    return links, n_links, n_inter_links, n_intra_links, n_non_as_links


def save_links_for_ases(node_ases, inter_node_links, file_logger, data_dir):
    count_links = {}
    inter_links = 0
    intra_links = 0
    non_as_links = 0
    links_to_write = []
    shuffle(inter_node_links)
    chunk_size = len(inter_node_links) // 8
    chunk_size = 1 if chunk_size == 0 else chunk_size
    partial_writer = partial(write_link_file, data_dir=data_dir)
    partial_parser = partial(link_parser, node_ases=node_ases)
    with Pool(8) as pool:
        for (
            links,
            n_links,
            n_inter_links,
            n_intra_links,
            n_non_as_links,
        ) in pool.imap_unordered(partial_parser, inter_node_links, chunk_size):
            inter_links += n_inter_links
            intra_links += n_intra_links
            non_as_links += n_non_as_links
            merge_dict(n_links, count_links)
            links_to_write.append(links)

    with Pool(8) as pool:
        for links in links_to_write:
            chunk_size = len(links) // 8
            chunk_size = 1 if chunk_size == 0 else chunk_size
            for _ in pool.imap_unordered(partial_writer, links.items(), chunk_size):
                pass
    msg = (
        "Links were saved, where: {} were ignored,".format(non_as_links)
        + " {} were inter AS".format(inter_links)
        + " {} were intra AS".format(intra_links)
    )
    file_logger.info(msg)
    return count_links


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
