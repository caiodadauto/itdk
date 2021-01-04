import os
import re

import numpy as np
import pandas as pd
from progress.counter import Counter

from itdk.logger import create_logger


def save(store, data, to_radians):
    ids = data["ids"]
    latitudes = data["latitudes"]
    longitudes = data["longitudes"]
    if to_radians:
        latitudes = np.radians(latitudes)
        longitudes = np.radians(longitudes)
    df = pd.DataFrame(
        {"id": ids, "latitude": latitudes, "longitude": longitudes},
        index=range(data["idxb"], data["idxe"]),
    )
    df.latitude = df.latitude.astype("float32")
    df.longitude = df.longitude.astype("float32")
    store.append("pandas", df, min_itemsize={"id": 9})


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


def check_buffers(stores, data_lists, to_radians):
    if type(stores) is dict:
        for key, data in data_lists.items():
            if len(data["ids"]) != 0:
                save(stores[key], data, to_radians)
    else:
        if len(data_lists["ids"]) != 0:
            save(stores, data_lists, to_radians)


def close_tables(stores, file_logger):
    for key, store in stores.items():
        file_logger.info(
            "The tabel for {} was closed\n{}".format(
                key, store.get_storer("pandas").table
            )
        )
        store.close()


# 38 countries
def hierarchical_list(geo_path, to_radians=False):
    stores = {}
    data_lists = {}
    counter = Counter("Processed Lines ")
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


def list(geo_path, to_radians=False):
    counter = Counter("Processed Lines ")
    file_logger = create_logger("location.log")
    os.makedirs("data/", exist_ok=True)
    store = pd.HDFStore("data/all_geo.h5")
    data = dict(ids=[], latitudes=[], longitudes=[], idxb=0, idxe=0)
    with open(geo_path, "r") as f:
        for line in f:
            if line[0] != "#":
                splited_line = re.split(r"\t", line)
                data["ids"].append(splited_line[0].split(" ")[1][:-1])
                data["latitudes"].append(
                    to_float(splited_line[5], file_logger, data["idxe"])
                )
                data["longitudes"].append(
                    to_float(splited_line[6], file_logger, data["idxe"])
                )
                data["idxe"] += 1

                if data["idxe"] - data["idxb"] == 100000:
                    save(store, data, to_radians)
                    data["idxb"] = data["idxe"]
                    data["ids"] = []
                    data["latitudes"] = []
                    data["longitudes"] = []
            counter.next()
        check_buffers(store, data, to_radians)
    file_logger.info(
        "The tabel for {} was closed".format(store.get_storer("pandas").table)
    )
    store.close()
