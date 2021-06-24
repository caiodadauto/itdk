import argparse

from itdk.geolocation_ases import process_location_with_ases


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("geo_path", type=str)
    p.add_argument("ases_path", type=str)
    p.add_argument(
        "--to-radians",
        action="store_true",
        help="Convert the latitude and longitude data from degrees to radians",
    )
    args = p.parse_args()
    process_location_with_ases(**vars(args))
