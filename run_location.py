import argparse

from itdk import location


def run(geo_location_path, as_location_path, to_radians, hierarchical):
    if hierarchical:
        location.hierarchical_list(geo_location_path, to_radians)
    else:
        location.list_with_ASes(geo_location_path, as_location_path, to_radians)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("geo_location_path", type=str)
    p.add_argument("as_location_path", type=str)
    p.add_argument(
        "--to-radians",
        action="store_true",
        help="Convert the latitude and longitude data from degrees to radians",
    )
    p.add_argument(
        "--hierarchical",
        action="store_true",
        help="List location for each country/region/city separately",
    )
    args = p.parse_args()
    run(**vars(args))
