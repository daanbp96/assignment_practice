import pandas as pd
from src.clean import clean_data
from src.features import generate_features
from src.models.random_forest import RandomForestModel
from src.models.baseline import train_baseline_model
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("RandomForest/Titanic/data/raw/train.csv")
test_df = pd.read_csv("RandomForest/Titanic/data/raw/test.csv")

# Clean data
cleaned_train_df = clean_data(train_df)
cleaned_test_df = clean_data(test_df)

# Feature engineering
eng_train_df = generate_features(cleaned_train_df)
eng_test_df = generate_features(cleaned_test_df)

# Split data
def split(df: pd.DataFrame, target: str = "Survived", test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# Split both cleaned and engineered
X_train_clean, X_val_clean, y_train_clean, y_val_clean = split(cleaned_train_df)
X_train_eng, X_val_eng, y_train_eng, y_val_eng = split(eng_train_df)

results = {}

# Random Forest on cleaned data
rf_clean = RandomForestModel("RF_clean")
rf_clean.train(X_train_clean, y_train_clean)
results['random_forest_clean'] = rf_clean.evaluate(X_val_clean, y_val_clean)

# Random Forest on engineered data
rf_eng = RandomForestModel("RF_eng")
rf_eng.train(X_train_eng, y_train_eng)
results['random_forest_engineered'] = rf_eng.evaluate(X_val_eng, y_val_eng)

# Save engineered training set
eng_train_df.to_csv("RandomForest/Titanic/data/processed/train.csv", index=False)

# Predict for submission (optional)
preds = rf_eng.predict(eng_test_df.drop(columns=["PassengerId"]))
submission = eng_test_df[["PassengerId"]].copy()
submission["Survived"] = preds
submission_path = "RandomForest/Titanic/data/results/submission_random_forest_engineered.csv"
submission.to_csv(submission_path, index=False)
print(f"✅ Saved predictions to {submission_path}")

# Print summary
print("\n=== Summary ===")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")
