from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

def split_train_test(
        df: pd.DataFrame,
        target_column: str,
        random_state: int
        ) -> list:
    X_val = df.drop(columns=[target_column])
    y_val = df[target_column]

    return train_test_split(
        X_val, y_val, test_size=0.2, stratify=y_val, random_state=random_state
        )