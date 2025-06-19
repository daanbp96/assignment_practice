from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from src.models.base import ModelWrapper

class RandomForestModel(ModelWrapper):
    def __init__(self):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)