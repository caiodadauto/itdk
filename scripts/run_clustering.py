import argparse

from itdk.cluster import run_clustering


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", default="data/clusters_lack_inters", type=str)
    p.add_argument("--graph-dir", default="data/raw_graphs_lack_inters", type=str)
    p.add_argument("--data-dir", default="data", type=str)
    p.add_argument("--min-graph-size", default=20, type=int)
    p.add_argument("--max-graph-size", default=60, type=int)
    p.add_argument("--n-threads", default=8, type=int)
    p.add_argument("--parallel", action="store_false")
    args = p.parse_args()
    run_clustering(**vars(args))
