import os
import re

import numpy as np
import pandas as pd
from progress.counter import Counter

from itdk.logger import create_logger
from itdk.ases import get_all_AS, save_ASes


def save(store, data, to_radians, use_as=False):
    ids = data["ids"]
    latitudes = data["latitudes"]
    longitudes = data["longitudes"]
    if to_radians:
        latitudes = np.radians(latitudes)
        longitudes = np.radians(longitudes)
    if use_as:
        df = pd.DataFrame(
            {
                "id": ids,
                "latitude": latitudes,
                "longitude": longitudes,
                "ases": data["ases"],
            },
            index=range(data["idxb"], data["idxe"]),
        )
    else:
        df = pd.DataFrame(
            {"id": ids, "latitude": latitudes, "longitude": longitudes},
            index=range(data["idxb"], data["idxe"]),
        )
    df.latitude = df.latitude.astype("float32")
    df.longitude = df.longitude.astype("float32")
    if use_as:
        store.append("geo", df, min_itemsize={"id": 9, "ases": 10})
    else:
        store.append("geo", df, min_itemsize={"id": 9})


def create_empty_lists():
    return dict(ids=[], latitudes=[], longitudes=[], idxb=0, idxe=0)


def create_store(country, region, city):
    os.makedirs("data/{}/{}/".format(country, region), exist_ok=True)
    return pd.HDFStore("data/{}/{}/{}_geo.h5".format(country, region, city))


def to_float(str_float, file_logger, index):
    try:
        v = float(str_float)
    except Exception:
        file_logger.info(
            "Not float format: {}, for index {}".format(str_float, index)
        )
        v = 360.0
    return v


def check_buffers(stores, data_lists, to_radians, use_as=False):
    if type(stores) is dict:
        for key, data in data_lists.items():
            if len(data["ids"]) != 0:
                save(stores[key], data, to_radians, use_as)
    else:
        if len(data_lists["ids"]) != 0:
            save(stores, data_lists, to_radians, use_as)


def close_tables(stores, file_logger):
    for key, store in stores.items():
        file_logger.info(
            "The tabel for {} was closed\n{}".format(
                key, store.get_storer("geo").table
            )
        )
        store.close()


# 38 countries
def hierarchical_list(geo_path, to_radians=False):
    stores = {}
    data_lists = {}
    counter = Counter("Geo Location Processed Lines ")
    file_logger = create_logger("location.log")
    with open(geo_path, "r") as f:
        for line in f:
            if line[0] != "#":
                splited_line = re.split(r"\t", line)
                country = splited_line[2] if splited_line[2] != "" else "NONE"
                region = splited_line[3] if splited_line[3] != "" else "NONE"
                city = splited_line[4] if splited_line[4] != "" else "NONE"
                key = country + ":" + region + ":" + city
                if key not in stores:
                    stores[key] = create_store(country, region, city)
                if key not in data_lists:
                    data_lists[key] = create_empty_lists()
                data_lists[key]["ids"].append(
                    splited_line[0].split(" ")[1][:-1]
                )
                data_lists[key]["latitudes"].append(
                    to_float(
                        splited_line[5], file_logger, data_lists[key]["idxe"]
                    )
                )
                data_lists[key]["longitudes"].append(
                    to_float(
                        splited_line[6], file_logger, data_lists[key]["idxe"]
                    )
                )
                data_lists[key]["idxe"] += 1

                if data_lists[key]["idxe"] - data_lists[key]["idxb"] == 10000:
                    save(stores[key], data_lists[key], to_radians)
                    data_lists[key]["ids"] = []
                    data_lists[key]["latitudes"] = []
                    data_lists[key]["longitudes"] = []
                    data_lists[key]["idxb"] = data_lists[key]["idxe"]
            counter.next()
        check_buffers(stores, data_lists, to_radians)
    close_tables(stores, file_logger)


def list_with_ASes(geo_path, as_file_path, to_radians=False):
    file_path = "data/itdk.h5"
    counter = Counter("Geo Location Processed Lines ")
    file_logger = create_logger("location.log")
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
            counter.next()
        check_buffers(store, data, to_radians, True)
    file_logger.info(
        "The tabel for {} was closed".format(store.get_storer("geo").table)
    )
    store.close()
