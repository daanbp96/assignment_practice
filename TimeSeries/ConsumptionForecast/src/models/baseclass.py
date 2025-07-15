from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import accuracy_score

class ModelWrapper(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        preds = self.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"[{self.model_name}] Validation Accuracy: {acc:.4f}")
        return acc
