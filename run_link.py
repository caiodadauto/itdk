import argparse

from itdk import links


def run(link_path, geo_ases_path):
    links.extract_links_for_ases(link_path, geo_ases_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("link_path", type=str)
    p.add_argument("geo_ases_path", type=str)
    args = p.parse_args()
    run(**vars(args))
