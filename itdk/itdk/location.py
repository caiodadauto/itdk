import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from itdk.logger import create_logger
from itdk.ases import get_all_AS, save_ASes


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
    store.append("geo", df, min_itemsize={"id": 9, "ases": 10})


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


def list_with_ASes(geo_path, as_file_path, to_radians=True):
    file_path = "data/geolocation_AS.h5"
    counter = tqdm("Processed lines")
    file_logger = create_logger("geolocation.log")
    os.makedirs("data/", exist_ok=True)
    store = pd.HDFStore(file_path)

    ASes = get_all_AS(as_file_path)
    save_ASes(ASes, file_path)
    data = dict(ids=[], latitudes=[], longitudes=[], ases=[], idxb=0, idxe=0)
    with open(geo_path, "r") as f:
        for line in f:
            if line[0] != "#":
                splited_line = re.split(r"\t", line)
                node_id = splited_line[0].split(" ")[1][:-1]
                if node_id in ASes:
                    data["ids"].append(node_id)
                    data["ases"].append(ASes[node_id])
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
    counter.close()
    file_logger.info(
        "The tabel for {} was closed".format(store.get_storer("geo").table)
    )
    store.close()
