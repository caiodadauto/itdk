import argparse

from itdk import ases


def run(hdf_path):
    ases.extract_topo_from_unique_positions(hdf_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("hdf_path", type=str)
    args = p.parse_args()
    run(**vars(args))
