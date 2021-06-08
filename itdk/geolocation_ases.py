import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from itdk.logger import create_logger
from itdk.ases import get_all_ases, save_ases


def save(store, data, to_radians):
    ids = data["ids"]
    latitudes = data["latitudes"]
    longitudes = data["longitudes"]
    if to_radians:
        latitudes = np.radians(latitudes)
        longitudes = np.radians(longitudes)
    df = pd.DataFrame(
        {
            "id": ids,
            "latitude": latitudes,
            "longitude": longitudes,
            "ases": data["ases"],
        },
        index=range(data["idxb"], data["idxe"]),
    )
    df.latitude = df.latitude.astype("float32")
    df.longitude = df.longitude.astype("float32")
    store.append("geo_with_ases", df, min_itemsize={"id": 9, "ases": 10})


def to_float(str_float, file_logger, index):
    try:
        v = float(str_float)
    except Exception:
        file_logger.info(
            "Not float format: {}, for index {}".format(str_float, index)
        )
        v = 360.0
    return v


def check_buffers(stores, data_lists, to_radians):
    if len(data_lists["ids"]) != 0:
        save(stores, data_lists, to_radians)


def close_tables(stores, file_logger):
    for key, store in stores.items():
        file_logger.info(
            "The tabel for {} was closed\n{}".format(
                key, store.get_storer("geo").table
            )
        )
        store.close()


def process_file(store, data, ases, path, counter, file_logger, to_radians):
    with open(path, "r") as f:
        for line in f:
            if line[0] != "#":
                splited_line = re.split(r"\t", line)
                node_id = splited_line[0].split(" ")[1][:-1]
                if node_id in ases:
                    data["ids"].append(node_id)
                    data["ases"].append(ases[node_id])
                    data["latitudes"].append(
                        to_float(splited_line[5], file_logger, data["idxe"])
                    )
                    data["longitudes"].append(
                        to_float(splited_line[6], file_logger, data["idxe"])
                    )
                    data["idxe"] += 1

                if data["idxe"] - data["idxb"] == 100000:
                    save(store, data, to_radians, True)
                    data["idxb"] = data["idxe"]
                    data["ids"] = []
                    data["ases"] = []
                    data["latitudes"] = []
                    data["longitudes"] = []
            counter.update()
        check_buffers(store, data, to_radians, True)

def process_location_with_ases(geo_path, ases_path, to_radians=True):
    log_dir = "logs"
    data_dir = "data"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "geolocation_with_ases.log")
    data_path = os.path.join(data_dir, "geolocation_with_ases.h5")
    file_logger = create_logger(log_path)
    store = pd.HDFStore(data_path)
    counter = tqdm("Processed lines [{}]".format(geo_path), leave=False)

    ases = get_all_ases(ases_path)
    save_ases(ases, data_path)
    data = dict(ids=[], latitudes=[], longitudes=[], ases=[], idxb=0, idxe=0)
    process_file(store, data, ases, data_path, counter, file_logger, to_radians)
    file_logger.info(
        "The tabel for {} was closed".format(store.get_storer("geo").table)
    )
    counter.close()
    store.close()
