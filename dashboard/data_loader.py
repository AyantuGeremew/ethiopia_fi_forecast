import pandas as pd
import os


def load_data(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    return df



def load_forecasts(file_path):

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    return df



def load_scenarios(file_path):

    return pd.read_csv(file_path)