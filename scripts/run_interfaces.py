import argparse

from itdk import interfaces


def run(path_to_ifaces, path_to_node_as, dir_name):
    interfaces.parse_interfaces(path_to_ifaces, path_to_node_as, dir_name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path_to_ifaces", type=str)
    p.add_argument("path_to_node_as", type=str)
    p.add_argument("--dir-name", type=str, default="interfaces")
    args = p.parse_args()
    run(**vars(args))
