import argparse

from itdk import graphs


def run(geo_path, link_path):
    graphs.create_graphs_from_ases(geo_path, link_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("geo_path", type=str)
    p.add_argument("link_path", type=str)
    args = p.parse_args()
    run(**vars(args))
