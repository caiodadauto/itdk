import argparse

from itdk.links import extract_links_for_ases


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("link_path", type=str)
    p.add_argument("geo_ases_path", type=str)
    args = p.parse_args()
    extract_links_for_ases(**vars(args))
