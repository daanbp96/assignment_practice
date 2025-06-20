from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from .base import ModelWrapper

class RandomForestModel(ModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict_proba(X), index=X.index)
