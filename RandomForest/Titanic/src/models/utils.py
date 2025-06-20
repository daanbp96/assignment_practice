# src/models/utils.py
from typing import NamedTuple
from sklearn.model_selection import train_test_split
import pandas as pd

class SplitData(NamedTuple):
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series

def split_train_test(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42) -> SplitData:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return SplitData(X_train, X_val, y_train, y_val)
