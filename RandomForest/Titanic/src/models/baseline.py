from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_baseline_model(X_train, y_train, strategy="most_frequent"):
    """
    Train a baseline DummyClassifier model using the specified strategy.

    Parameters:
    - df: pandas DataFrame containing features and target
    - target_column: str, name of the target column in df
    - strategy: str, strategy used by DummyClassifier ('most_frequent' by default)

    Returns:
    - model: trained DummyClassifier
    - accuracy: accuracy score on test split
    """
    model = DummyClassifier(strategy=strategy, random_state=42)
    model.fit(X_train, y_train)
    return model
