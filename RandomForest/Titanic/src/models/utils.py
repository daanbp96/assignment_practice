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
        ) # type: ignore


def evaluate_model(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "model"
) -> float:
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"[{model_name}] Validation Accuracy: {acc:.4f}")
    return acc

def make_predictions(
    model: RandomForestClassifier,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predicts the target on test data and prepares submission DataFrame.
    
    Args:
        model: Trained RandomForestClassifier.
        test_df: Test features DataFrame.
        
    Returns:
        DataFrame with 'PassengerId' and 'Survived' columns ready for submission.
    """
    preds = model.predict(test_df)
    submission = test_df[['PassengerId']].copy()
    submission['Survived'] = preds
    return submission