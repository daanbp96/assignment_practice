from sklearn.ensemble import RandomForestClassifier
from RandomForest.Titanic.src.models.utils import split_train_test

import pandas as pd

def train_random_forest(
        df: pd.DataFrame,
        random_state: int = 42
) -> tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    
    X_train, X_val, y_train, y_val = split_train_test(df, target_column = 'Survived')
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model
