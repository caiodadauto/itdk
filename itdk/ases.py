import re

import pandas as pd
from tqdm import tqdm


def save_ases(ases, file_path):
    pd.DataFrame(set(ases.values()), columns=["ases"]).to_hdf(
        file_path, "ases", mode="a", min_itemsize={"ases": 10}
    )

def get_all_ases(file_path):
    ases = {}
    counter = tqdm("Processed lines [{}]".format(file_path))
    with open(file_path, "r") as f:
        for line in f:
            if line[0] != "#":
                splited_line = re.split(r"\s", line)
                node_id = splited_line[1]
                as_name = splited_line[2]
                ases[node_id] = as_name
            counter.update()
    counter.close()
    return ases
