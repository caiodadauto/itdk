import os
import re

import numpy as np
import pandas as pd
from progress.counter import Counter

from itdk.logger import create_logger


def search_query(ids):
    query = ""
    for node_id in ids[:-1]:
        query += "id=='{}' or ".format(node_id)
    query += "id=='{}'".format(ids[-1])
    return query


def save_interfaces(data, all_node_as, dirpath, file_logger):
    discarted_nodes = []
    df = pd.DataFrame(data, columns=["addrs", "id"])
    df.sort_values("id", inplace=True)
    node_ids = np.unique(df["id"].values)
    for node_id in node_ids:
        try:
            ases = all_node_as.loc[node_id].values
        except KeyError:
            discarted_nodes += node_id
            continue
        node_interfaces = df.loc[df["id"] == node_id]
        for as_name in ases:
            file_path = os.path.join(dirpath, "{}.csv".format(as_name))
            node_interfaces.to_csv(file_path, header=False, index=False, mode="a")
    if len(discarted_nodes) != 0:
        file_logger.info(
                "Node(s) {} without geolocation".format(discarted_nodes))


def process(inter_path, file_logger, node_as_path, dirpath, node_as, buffer_size=500000):
    data = np.zeros((buffer_size, 2), dtype='<U15')
    regular = re.compile(r"N+\d")
    counter = Counter("Processed Interfaces ")
    saved_lines = 0
    with open(inter_path, "r") as f:
        for line in f:
            if line[0] != "#":
                splited_line = re.split("\s", line)
                size = len(splited_line)
                if size > 1 and regular.match(splited_line[1]):
                    data[saved_lines] = splited_line[:2]
                    saved_lines += 1
                    if saved_lines % buffer_size == 0:
                        save_interfaces(data, node_as, dirpath, file_logger)
                        data = np.zeros((buffer_size, 2), dtype='<U15')
                        saved_lines = 0
            counter.next()
    if saved_lines > 0:
        save_interfaces(data, node_as, dirpath, file_logger)


def parse_interfaces(inter_path, node_as_path, dirname):
    dirpath = os.path.join("data", dirname)
    os.makedirs(dirpath)
    file_logger = create_logger("interfaces.log")
    node_as = pd.read_hdf(node_as_path, "geo", columns=["id", "ases"])
    node_as.set_index("id", inplace=True)
    process(inter_path, file_logger, node_as_path, dirpath, node_as)
