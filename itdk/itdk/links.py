import os
import re
import gzip

from progress.counter import Counter

from itdk.logger import create_logger


def get_nodes_from_link(line):
    neighborhood = []
    splited_line = re.split(r"\s", line)
    node = splited_line[3].split(":")[0]
    for raw_node in splited_line[4:-2]:
        neighborhood.append(raw_node.split(":")[0])
    return node, neighborhood


def is_empty(root_dir):
    files = [
        name for name in os.listdir(root_dir) if name.split(".")[-1] == "gz"
    ]
    return len(files) == 0


def save_links(links, root_dir):
    for node, neighborhood in links.items():
        with gzip.GzipFile(
            os.path.join(root_dir, node + ".txt.gz"), "ab"
        ) as file:
            content = ""
            for neighbor in neighborhood:
                content += neighbor + "\n"
            file.write(content.encode())


def get_edges(link_path):
    links = {}
    root_dir = "data/links/"
    counter = Counter("Processed Links ")
    file_logger = create_logger("link.log")
    os.makedirs(root_dir, exist_ok=True)
    if not is_empty(root_dir):
        msg = "The directory {} is not empty.".format(root_dir)
        file_logger.error(msg)
        raise IOError(msg)
    file_logger.info(
        "Start the link extraction from {}".format(os.path.abspath(link_path))
    )
    with open(link_path, "r") as f:
        for line in f:
            if line[0] != "#":
                node, neighborhood = get_nodes_from_link(line)
                if node in links:
                    links[node] += neighborhood
                else:
                    links[node] = neighborhood
                if len(links) == 100000:
                    save_links(links, root_dir)
                    links = {}
                counter.next()
    if len(links) > 0:
        save_links(links, root_dir)
    file_logger.info(
        "The links was extracted and saved in {}".format(
            os.path.abspath(root_dir)
        )
    )
