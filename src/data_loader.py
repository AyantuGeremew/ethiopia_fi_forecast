import os
import openpyxl
import pandas as pd


def load_dataset(file_path: str, sheet_name=0):
    """
    Load an Excel sheet into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to Excel file.
    sheet_name : int or str
        Sheet index or sheet name.

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✓ Dataset loaded successfully.")
        return df

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def load_reference_data(file_path):
    """
    Load reference codes excel
    """

    return pd.read_excel(file_path)