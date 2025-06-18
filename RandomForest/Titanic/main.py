import pandas as pd
from src.clean import clean_data
from src.features import generate_features
from src.models.random_forest import train_random_forest
from src.models.baseline import train_baseline_model
from RandomForest.Titanic.src.models.utils import evaluate_model, make_predictions, split_train_test
from src.predict import make_predictions

# Load data
train_df = pd.read_csv("RandomForest/Titanic/data/raw/train.csv")
test_df = pd.read_csv("RandomForest/Titanic/data/raw/test.csv")

# Clean data
cleaned_train_df = clean_data(train_df)
cleaned_test_df = clean_data(test_df)

eng_trained_df = generate_features(cleaned_train_df)
eng_test_df = generate_features(cleaned_test_df)


# Train/val split
X_train, X_val, y_train, y_val = split_train_test(cleaned_train_df, target_column = 'Survived')


results = {}

# Baseline model
baseline_model = train_baseline_model(X_train, y_train)
results['baseline'] = evaluate_model(baseline_model, X_val, y_val, "baseline")

# RF no engineered features
rf_model_no_feat = train_random_forest(cleaned_train_df)
results['random_forest_no_features'] = evaluate_model(rf_model_no_feat, X_val, y_val, "random_forest_no_features")

rf_model_feat = train_rf(X_train_eng, y_train_eng)
results['random_forest_with_features'] = evaluate_model(rf_model_feat, X_val_eng, y_val_eng, "random_forest_with_features")

# Save engineered data for inspection
eng_train_df.to_csv("RandomForest/Titanic/data/processed/train.csv", index=False)

# Predict on test data
submission = make_predictions(rf_model_feat, eng_test_df)
submission_path = "RandomForest/Titanic/data/results/submission_random_forest_with_features.csv"
submission.to_csv(submission_path, index=False)
print(f"✅ Saved predictions to {submission_path}")

print("\n=== Summary ===")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")
