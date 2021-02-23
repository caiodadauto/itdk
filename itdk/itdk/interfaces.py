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


def save_interfaces(data, node_as_path, dirpath, file_logger):
    df = pd.DataFrame(data, columns=["addrs", "id"])
    df.sort_values("id", inplace=True)
    query = search_query(np.unique(df["id"].values))
    node_as = pd.read_hdf(node_as_path, "geo", columns=["id", "ases"], where=query)
    node_as.sort_values("id", inplace=True)
    if node_as.shape[0] < df.shape[0]:
        size = node_as.shape[0]
        indices = node_as["id"].searchsorted(df["id"])

        mask = indices < size
        discarted_nodes = df["id"].loc[~mask].values
        df = df.loc[mask]
        indices = indices[mask]

        mask = node_as["id"].iloc[indices].values == df["id"].values
        discarted_nodes = np.concatenate([discarted_nodes, df["id"].loc[~mask].values])
        df = df.loc[mask]
        indices = indices[mask]
        node_as = node_as.iloc[indices]
        if discarted_nodes.size != 0:
            file_logger.info(
                    "Node(s) {} without geolocation".format(discarted_nodes))
    ases = node_as["ases"].values
    df = df.assign(ases=ases)
    as_names = np.unique(ases)
    for as_name in as_names:
        file_path = os.path.join(dirpath, "{}.csv".format(as_name))
        as_df = df.loc[df["ases"] == as_name, ("addrs", "id")]
        as_df.to_csv(file_path, header=False, index=False, mode="a")


def process(inter_path, file_logger, node_as_path, dirpath, buffer_size=30):
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
                        save_interfaces(data, node_as_path, dirpath, file_logger)
                        data = np.zeros((buffer_size, 2), dtype='<U15')
                        saved_lines = 0
            counter.next()
    if saved_lines > 0:
        save_interfaces(data, node_as_path, dirpath, file_logger)


def parse_interfaces(inter_path, node_as_path, dirname):
    dirpath = os.path.join("data", dirname)
    os.makedirs(dirpath)
    file_logger = create_logger("interfaces.log")
    process(inter_path, file_logger, node_as_path, dirpath)
