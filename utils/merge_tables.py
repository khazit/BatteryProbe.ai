"""Reads CSV files from a directory and merge them into one single table."""


import os
import sys

import pandas as pd
from tqdm import tqdm


def read_feature(data_dir, feature_name):
    """Read a specific CSV file given a data directory and a feature name."""
    # Read csv
    data = pd.read_csv(f"{data_dir}/{feature_name}.csv", low_memory=False)

    # Convert epoch to seconds
    data["time"] = data["time"] / 10e8

    # Merge 'battery_status' and 'Battery_status'
    if "Battery_status" in data:
        data["battery_status"] = data["battery_status"]\
            .fillna(data["Battery_status"])
        data = data.drop("Battery_status", axis=1)

    # Rename 'value'
    data = data.rename(columns={"value": feature_name})

    # Drop 'name' column
    data = data.drop("name", axis=1)
    return data


if __name__ == "__main__":
    data_dir = sys.argv[1]
    assert os.path.isdir(data_dir), "Data directory not found"
    out_file = os.path.join(data_dir, "all.csv")
    if os.path.isfile(out_file):
        os.remove(out_file)

    # Read all the csv files
    dataframes = []
    print("Reading CSV files ...")
    for feature in tqdm(os.listdir("data_54.38.188.95/")):
        data = read_feature(data_dir, feature.split(".")[0])
        dataframes.append(data)

    # Merge everything
    data = dataframes[0]
    print("Merging tables ...")
    for df in tqdm(dataframes[1:]):
        data = data.merge(
            df, how="outer",
            on=["time", "battery_status", "manufacturer", "os", "uuid"],
        )

    # Merge "fans_rpm" and "mean_fans_rpm"
    data["fans_rpm"] = data["fans_rpm"].fillna(data["mean_fans_rpm"])

    # Write big table to disk
    print("Writing table to disk ...")
    data.to_csv(out_file)
    print("Done")
    sys.exit(0)
