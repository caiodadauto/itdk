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


def save_interfaces(data, node_as_path, dirpath):
    df = pd.DataFrame(data, columns=["addrs", "id"])
    df.sort_values("id")
    query = search_query(np.unique(df["id"].values))
    node_as = pd.read_hdf(node_as_path, "geo", columns=["id", "ases"], where=query)
    node_as.sort_values("id")
    ases = node_as["ases"].values
    df["as"] = ases
    as_names = np.unique(ases)
    for as_name in as_names:
        file_path = os.path.join(dirpath, "{}.csv".format(as_name))
        as_df = df.loc[df["as"] == as_name, ("addrs", "id", "as")]
        as_df.to_csv(file_path, index=False, mode="a")


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
                        save_interfaces(data, node_as_path, dirpath)
                        data = np.zeros((buffer_size, 2))
                        saved_lines = 0
            counter.next()
    if saved_lines > 0:
        save_interfaces(data, node_as_path, dirpath)


def parse_interfaces(inter_path, node_as_path, dirname):
    dirpath = os.path.join("data", dirname)
    os.makedirs(dirpath)
    file_logger = create_logger("interfaces.log")
    process(inter_path, file_logger, node_as_path, dirpath)
