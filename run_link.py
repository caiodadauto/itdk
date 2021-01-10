import argparse

from itdk import links


def run(link_path):
    links.get_edges(link_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("link_path", type=str)
    args = p.parse_args()
    run(**vars(args))
