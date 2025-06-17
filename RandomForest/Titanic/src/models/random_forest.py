from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def split_data(
    df: pd.DataFrame,
    target_col: str = 'Survived',
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def train_model(
    df: pd.DataFrame,
    target_col: str = 'Survived',
    random_state: int = 42
) -> tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    X_train, X_val, y_train, y_val = split_data(df, target_col, random_state=random_state)
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_val, y_val


def evaluate_model(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "random_forest"
) -> float:
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"[{model_name}] Validation Accuracy: {acc:.4f}")
    return acc
