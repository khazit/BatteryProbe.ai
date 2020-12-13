"""Define the input data pipeline."""

import random
import logging

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from batteryprobe.utils import pad_and_pack


class SessionDataset(Dataset):
    """
    Session dataset.

    Attributes:
        lower_bound (float): Lower bound when splitting the labels from the inputs.
        upper_bound (float): Upper bound when splitting the labels from the inputs.
        features (list): List of features to use.
        context (list): List of contexts to use.

    Args:
        sessions (list): List of pd.DataFrame sessions.
        params (dict): Parameters dict.
    """

    def __init__(self, sessions, params):
        self.sessions = sessions
        self.lower_bound = params["label_lower_bound"]
        self.upper_bound = params["label_upper_bound"]
        self.features = params["features"]
        self.context = params["context"]

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        # Generate random float between 20-50%
        rand = random.uniform(self.lower_bound, self.upper_bound)

        # Take the breakpoint index
        middlepoint = int(rand * len(self.sessions[idx]))

        # Cut into inputs and labels
        data = self.sessions[idx]
        inputs = data[:middlepoint][self.features + self.context]
        time = data["time"]
        context = data.iloc[middlepoint:][self.context]
        labels = data.iloc[middlepoint:][self.features]
        return (inputs.values, time.values, context.values), labels.values


def _extract_sessions(data, params):
    """Extract sessions.

    Args:
        data (pd.DataFrame): data from one unique user (uuid
            should be the same for all rows)
        params (dict): parameters dict

    Returns:
        List of sessions (pd.Dataframes)
    """
    if len(data) <= 1:
        return []

    logging.debug(f"{data['uuid'].iloc[0]} - {len(data)} points")
    sessions = []

    while len(data) != 0:
        # Initialize some variables
        session = []
        is_charging = data.iloc[0]["battery_status_Charging"]
        epoch = data.iloc[0]["time"]
        index = data.iloc[0].name
        session.append(data.iloc[0])

        # A session is ended with a:
        # * battery status change (from discharging to charging for eg)
        # * long enough gap between two consecutive data points
        for index_sess, row_sess in data[~data.index.isin([index])].iterrows():
            last_index = index_sess
            if row_sess["battery_status_Charging"] != is_charging:
                break
            if row_sess["time"] - epoch > params["max_gap_between_sessions"]:
                break
            session.append(row_sess)
            epoch = row_sess["time"]

        # Remove all rows that were processed
        data = data[data.index > last_index]
        # Add session if it's long enough (w.r.t time and # of points)
        if (len(session) > params["min_num_points_per_session"]) and \
                (session[-1]["time"] - session[0]["time"]) > params["min_session_duration"]:
            # Add session to list (w/o uuid)
            sessions.append(session)

    logging.debug(f"Extracted {len(sessions)} sessions")
    sessions_df = [pd.DataFrame(session).drop(["uuid"], axis=1) for session in sessions]
    return sessions_df


def _preprocess(data, params):
    """Preprocess dataframe."""
    # Dict to store some values that we'll store in disk later
    artefacts = {}

    # Keep points where battery is either charging or discharging
    data = data.loc[data["battery_status"].isin(["Charging", "Discharging"])]

    # Drop points where "charge_now" is nan
    data = data.dropna(subset=["charge_now", "charge_full"])
    data["capacity"] = data["capacity"]
    data["capacity"] = data["capacity"].fillna(100*data["charge_now"] / data["charge_full"])

    # Convert categorical features to one-hot representation
    battery_status = pd.get_dummies(data['battery_status'], prefix='battery_status')
    os_cat = pd.get_dummies(data['os'], prefix='os')
    data = pd.concat([data, battery_status, os_cat], axis=1)

    # Drop obsolete features
    data = data.drop(['battery_status', 'mean_fans_rpm', 'manufacturer', 'os'], axis=1)

    # Sort valus in chronological order
    data = data.sort_values(["time"])

    # Normalize data
    features_to_normalize = [
        "cpu_speed",
        "n_running_threads",
        "cycle_count",
        "ram_load",
        "current_now",
        "charge_full",
        "load_average_15",
        "fans_rpm",
        "voltage_min_design",
        "cpu_temp",
        "battery_temp",
        "load_average_1",
        "swap_load",
        "voltage_now",
        "load_average_5",
        "charge_full_design",
    ]
    artefacts["normalized_features"] = features_to_normalize
    artefacts["mean"] = data[features_to_normalize].mean()
    artefacts["std"] = data[features_to_normalize].std()
    data[features_to_normalize] = (
        data[features_to_normalize] - artefacts["mean"]) / artefacts["std"]

    # Convert epoch to sin
    period = 60 * 60
    data["epoch_sin"] = np.sin(data["time"] * (2 * np.pi / period))
    data["epoch_cos"] = np.cos(data["time"] * (2 * np.pi / period))
    # Drop remaining NaN rows (IMPORTANT: this should always be at the end)
    data = data.dropna(subset=params["features"]+params["context"])

    return data, artefacts


def collate_fn(batch):
    """Groups element of the dataset into a single batch."""
    return (
        (
            pad_and_pack([el[0][0] for el in batch]),  # inputs
            pad_and_pack([el[0][1] for el in batch]),  # time
            pad_and_pack([el[0][2] for el in batch]),  # context
        ),
        pad_and_pack([el[1] for el in batch]),
    )


def create_data_loader(data, params):
    """Create two (train/val) pytorch data loaders."""
    # Preprocess dataframe
    data, _ = _preprocess(data, params)
    logging.info(f"{len(data)} data points")

    # Search for NaN values
    features = params["features"]
    if data[features].isnull().values.any():
        logging.warning("Found NaN values in " + ", ".join(
            data[features].columns[data[features].isna().any()].tolist())
                        )

    # Extract sessions
    sessions_df = []
    for uuid in data["uuid"].unique():
        subsample = data[data["uuid"] == uuid]
        sessions_df += _extract_sessions(subsample, params)
    logging.info(f"Extracted {len(sessions_df)} sessions")

    # Split into train/val
    random.shuffle(sessions_df)
    split_breakpoint = int(len(sessions_df) * params["train_split"])
    if params["debug"]:
        train_sessions = [sessions_df[0]] * 200
    else:
        train_sessions = sessions_df[:split_breakpoint]
    train_sessions = train_sessions * params["repeat"]
    val_sessions = sessions_df[split_breakpoint:]

    # Create train SessionDataset
    train_ds = SessionDataset(train_sessions, params)
    train_ds = DataLoader(
        train_ds, collate_fn=collate_fn,
        batch_size=params["batch_size"], shuffle=True,
        num_workers=params["n_data_workers"],
        prefetch_factor=params["prefetch"],
    )

    # Create val SessionDataset
    val_ds = SessionDataset(val_sessions, params)
    val_ds = DataLoader(
        val_ds, collate_fn=collate_fn,
        batch_size=params["batch_size"], shuffle=True,
    )
    return train_ds, val_ds
