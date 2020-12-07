"""Define the input data pipeline."""


import random
import logging

import numpy as np
import pandas as pd
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class SessionDataset(Dataset):
    """Session dataset.

    Attributes:
        lower_bound (float): Lower bound when splitting the labels from the inputs.
        upper_bound (float): Upper bound when splitting the labels from the inputs.
        features (list): List of features to use.

    Args:
        sessions (list): List of pd.DataFrame sessions.
        params (dict): Parameters dict
    """
    def __init__(self, sessions, params):
        self.sessions = sessions
        self.lower_bound = params["label_lower_bound"]
        self.upper_bound = params["label_upper_bound"]
        self.features = params["features"]

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        # Generate random float between 20-50%
        rand = random.uniform(self.lower_bound, self.upper_bound)
        # Take the breakpoint index
        middlepoint = int(rand * len(self.sessions[idx]))

        # Cut into inputs and labels
        data = self.sessions[idx][self.features]
        inputs = data[:middlepoint]
        labels = data.iloc[middlepoint] # TODO: change
        return (inputs.values, labels.values)


def _extract_sessions(data, params):
    """Extract sessions.

    Args:
        data (pd.DataFrame): data from one unique user (uuid
            should be the same for all rows)
        params (dict): parameters dict

    Returns:
        List of sessions (pd.Dataframes)
    """
    sessions = []

    logging.debug(f"{data['uuid'].iloc[0]} - {len(data)} points")

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

    # Drop points where "charge_now" is nan
    data = data.dropna(subset=["charge_now"])

    # Keep points where battery is either charging or discharging
    data = data.loc[data["battery_status"].isin(["Charging", "Discharging"])]

    # Convert categorical features to one-hot representation
    battery_status = pd.get_dummies(data['battery_status'], prefix='battery_status')
    os_cat = pd.get_dummies(data['os'], prefix='os')
    data = pd.concat([data, battery_status, os_cat], axis=1)

    # Drop obsolete features
    data = data.drop(['battery_status', 'mean_fans_rpm', 'manufacturer', 'os'], axis = 1)

    # Sort valus in chronological order
    data = data.sort_values(["time"])

    # Convert capacity to float
    data["capacity"] = data["capacity"] / 100

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
        "charge_now",
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
    day = 24*60*60
    data["epoch_sin_day"] = np.sin(data["time"] * (2 * np.pi / day))

    # Drop remaining NaN rows (IMPORTANT: this should always be at the end)
    data = data.dropna(subset=params["features"])

    return data, artefacts


def collate_fn(batch):

    def pad_and_pack(idx):
        lengths = [el[idx].shape[0] for el in batch]
        data  = [from_numpy(el[idx]) for el in batch]
        data = pad_sequence(data, batch_first=True, padding_value=-999)
        data = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        return data
    import torch # TODO: change
    return (pad_and_pack(0), torch.stack([from_numpy(el[1]) for el in batch])) ## TODO: ch



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
    train_sessions = sessions_df[:int(len(sessions_df) * params["train_split"])]
    val_sessions = sessions_df[int(len(sessions_df) * params["train_split"]):]

    train_ds = SessionDataset(train_sessions, params)
    train_ds = DataLoader(
        train_ds, collate_fn=collate_fn,
        batch_size=params["batch_size"], shuffle=True,
    )
    val_ds = SessionDataset(val_sessions, params)
    val_ds = DataLoader(
        val_ds, collate_fn=collate_fn,
        batch_size=params["batch_size"], shuffle=False,
    )
    return train_ds, val_ds
