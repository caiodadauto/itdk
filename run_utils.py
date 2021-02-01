import argparse

from itdk import utils


def run(path_to_ases_nums, path_to_draw):
    utils.get_dist_nodes(path_to_ases_nums, path_to_draw)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path_to_ases_nums", type=str)
    p.add_argument("path_to_draw", type=str)
    args = p.parse_args()
    run(**vars(args))
