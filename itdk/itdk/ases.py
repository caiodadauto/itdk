import re

from progress.counter import Counter


def get_all_AS(file_path):
    ASes = {}
    counter = Counter("AS Processed Lines ")
    with open(file_path, "r") as f:
        for line in f:
            if line[0] != "#":
                splited_line = re.split(r"\s", line)
                node_id = splited_line[1]
                as_name = splited_line[2]
                ASes[node_id] = as_name
            counter.next()
    return ASes
