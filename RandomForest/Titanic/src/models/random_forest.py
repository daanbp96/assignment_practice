from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from .base import ModelWrapper

class RandomForestModel(ModelWrapper):
    def __init__(self, model_name: str, n_features_to_select: int = 6):
        super().__init__(model_name)
        self.n_features_to_select = n_features_to_select
        self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=1)
        self.selected_features = None
        self.rfe = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        # Run RFE to select top features
        self.rfe = RFE(estimator=RandomForestClassifier(random_state=42), 
                       n_features_to_select=self.n_features_to_select)
        self.rfe.fit(X, y)

        # Store selected feature names
        self.selected_features = X.columns[self.rfe.support_]

        # Fit the model on selected features only
        self.model.fit(X[self.selected_features], y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Use only selected features for prediction
        X_sel = X[self.selected_features]
        preds = self.model.predict(X_sel)
        return pd.Series(preds, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X_sel = X[self.selected_features]
        proba = self.model.predict_proba(X_sel)
        return pd.DataFrame(proba, index=X.index)

    def cross_val_score(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []
        feature_importances = []

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Run RFE per fold to avoid leakage
            rfe = RFE(estimator=RandomForestClassifier(random_state=42), 
                      n_features_to_select=self.n_features_to_select)
            rfe.fit(X_train, y_train)
            selected_feats = X_train.columns[rfe.support_]

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train[selected_feats], y_train)
            preds = model.predict(X_val[selected_feats])

            acc = accuracy_score(y_val, preds)
            accuracies.append(acc)
            feature_importances.append(model.feature_importances_)

        mean_accuracy = np.mean(accuracies)
        importances_df = pd.DataFrame(feature_importances, columns=selected_feats)
        mean_importances = importances_df.mean().sort_values(ascending=False)

        print(f"Mean Accuracy (top {self.n_features_to_select} features): {mean_accuracy:.4f}")
        print(mean_importances)

        return mean_accuracy, mean_importances
