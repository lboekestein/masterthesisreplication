"""
Auxiliary functions for the thesis project.

Contains:
- Data retrieving functions:
    - retrieve_data_from_api: retrieves data from the API and returns as a pandas DataFrame
    - load_queryset: loads a queryset from the data folder and returns as a pandas DataFrame
- Date functions:
    - date_to_month_id: converts year and month to month_id
    - month_id_to_ym: converts month_id to month name and year
"""

from typing import Optional
import pandas as pd
import calendar
import os


### --------------------------###
### Data retrieving functions ###
### --------------------------###

def load_queryset(
        queryset: str, 
        data_path: Optional[str] = "../data/raw/",
        type: str = "parquet"
    ) -> pd.DataFrame:
    """
    Load a queryset from the data folder.
    If the file does not exist, raise a FileNotFoundError.

    Args:
        queryset (str): name of the queryset to load (e.g. fatalities002_conflict_history)
        data_path (optional, str): path to the data folder (default is "../data/raw/")
        type (str): file type to load (default is "parquet")
    Returns:
        pd.DataFrame: DataFrame containing the queryset data
    """
    path = f"{data_path}{queryset}.{type}"

    if os.path.exists(path):
        print(f"Loading {queryset} from local storage...")
        if type == "parquet":
            return pd.read_parquet(path)
        elif type == "csv":
            return pd.read_csv(path)
        elif type == "pkl":
            return pd.read_pickle(path)
        else:
            raise ValueError(f"Unsupported file type: {type}")
    else:
        raise FileNotFoundError(f"{queryset} not found in local storage. Please ensure the file exists at {path}.")
    

### --------------------------###
### Date converting functions ###
### --------------------------###
    

def date_to_month_id(year: int, month: int) -> int:
    """ 
    Convert year and month to month_id.

    Args:
        year (int): year (e.g. 2022)
        month (int): month (1-12)
    
    Returns:
        int: month_id
    """

    return (year - 1980) * 12 + month


def month_id_to_ym(month_id: int) -> str:
    """
    Converts month_id to month name and year

    Args:
        month_id: integer representing the month id
    
    Returns:
        String consisting of month name and year
    """

    offset = month_id - 1
    year = 1980 + offset // 12
    month_num = offset % 12 + 1
    month_name = calendar.month_name[month_num]

    return f"{month_name} {year}"