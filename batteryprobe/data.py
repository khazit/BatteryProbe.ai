""""""


import random
import logging 

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


MAX_GAP_BETWEEN_SESSIONS = 400  # in seconds
MIN_NUM_POINTS_PER_SESSION = 10
MIN_SESSION_DURATION = 600 # in seconds


class SessionDataset(Dataset):
    """"""
    def __init__(self, sessions, params):
        self.sessions = sessions
        self.lower_bound = params["lower_bound"]
        self.upper_bound = params["upper_bound"]
        self.features = params["features"]

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        # Generate random number between 20-50%
        rand = random.uniform(self.lower_bound, self.upper_bound)
        # Take the breakpoint number
        breakpoint = int(rand * len(self.sessions[idx]))
        # Cut into inputs and labels
        data = self.sessions[idx][self.features]
        inputs = data[:breakpoint]
        labels = data[breakpoint:]
        return (inputs.values, labels.values)


def extract_sessions(data):
    """Extract sessions.
    
    Args: 
        data (pd.DataFrame): data from one unique user (uuid 
            should be the same for all rows)
            
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
            elif row_sess["time"] - epoch > MAX_GAP_BETWEEN_SESSIONS:
                break
            else:
                session.append(row_sess)
            epoch = row_sess["time"]

        # Remove all rows that were processed
        data = data[data.index > last_index]
        # Add session if it's long enough (w.r.t time and # of points)
        if (len(session) > MIN_NUM_POINTS_PER_SESSION) and \
          (session[-1]["time"] - session[0]["time"]) > MIN_SESSION_DURATION:
            # Add session to list (w/o uuid)
            sessions.append(session)

    logging.debug(f"Extracted {len(sessions)} sessions")
    sessions_df = [pd.DataFrame(session).drop(["uuid"], axis=1) for session in sessions]
    return sessions_df


def preprocess(data):
    """"""
    data = data.loc[data["battery_status"].isin(["Charging", "Discharging"])]
    data.loc[:, "fans_rpm"] = data["fans_rpm"].fillna(data["mean_fans_rpm"])
    battery_status = pd.get_dummies(data['battery_status'], prefix='battery_status')
    os = pd.get_dummies(data['os'], prefix='os')
    data = pd.concat([data, battery_status, os], axis=1)
    data = data.drop(['battery_status', 'mean_fans_rpm', 'manufacturer', 'os'], axis = 1) 
    data = data.sort_values(["time"])
    return data


def create_data_loader(data, params):
    """"""
    data = preprocess(data)
    sessions_df = []
    for uuid in data["uuid"].unique():
        subsample = data[data["uuid"] == uuid]    
        sessions_df += extract_sessions(subsample)
    dataset = SessionDataset(sessions_df, params)
    dataloader = DataLoader(
        dataset,
        batch_size=1, shuffle=True,
    )
    return dataloader
