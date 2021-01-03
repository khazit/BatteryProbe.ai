"""Script used to run experiments from one or more TOML config files."""


import sys
import uuid
import logging
from os.path import isfile, join

import toml
import torch
import pandas as pd

from batteryprobe.data import create_data_loader
from batteryprobe.loops import train, evaluate
from batteryprobe.models import Baseline, AutoRegressive


logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    for filepath in sys.argv[1:]:
        assert isfile(filepath), f"{filepath} is not a file."
        logging.info(f"Loading parameters from {filepath}")
        params = toml.load(filepath)
        params["log_dir"] = join(params["log_dir"], uuid.uuid4().hex)

        # Data
        logging.info(f"Reading and processing data from {params['data_path']}")
        df = pd.read_csv("data_54.38.188.95/all.csv")
        df = df.iloc[::params["skip_frequency"]]
        train_dls, val_dl = create_data_loader(df, params)
        target_col = params["features"].index("capacity")

        # Baseline
        logging.info("Evaluating baseline model")
        baseline = Baseline(target_col)
        base_score = evaluate(baseline, val_dl, target_col)

        # Training
        logging.info("Training model")
        model = AutoRegressive(params)
        train(model, (train_dls, val_dl), params)

        # Evaluation
        logging.info("Evaluating trained model")
        if not params["debug"]:
            model.load_state_dict(torch.load("model.pt"))
            model.eval()
            score = evaluate(model, val_dl, target_col=target_col)
        else:
            from batteryprobe.utils import plot_sample
            plot_sample(train_dls[0], target_col, n=1, model=model)
