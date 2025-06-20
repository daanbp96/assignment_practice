import pandas as pd

from src.clean import clean_data
from src.features import generate_features
from src.models.random_forest import RandomForestModel
from src.models.utils import split_train_test, SplitData

# === Constants ===
TARGET_COLUMN = "Survived"
PREDICT_SUBMISSION = False

TRAIN_PATH = "RandomForest/Titanic/data/raw/train.csv"
TEST_PATH = "RandomForest/Titanic/data/raw/test.csv"
PROCESSED_TRAIN_PATH = "RandomForest/Titanic/data/processed/train.csv"
SUBMISSION_PATH = "RandomForest/Titanic/data/results/submission_random_forest_engineered.csv"


def run():
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Clean data
    cleaned_train_df = clean_data(train_df)
    cleaned_test_df = clean_data(test_df)

    # Feature engineering
    eng_train_df = generate_features(cleaned_train_df)
    eng_test_df = generate_features(cleaned_test_df)

    # Split cleaned and engineered
    split_clean = split_train_test(cleaned_train_df, TARGET_COLUMN)
    split_eng = split_train_test(eng_train_df, TARGET_COLUMN)

    # Train and evaluate models
    results = {}

    rf_clean = RandomForestModel("RF_clean")
    rf_clean.train(split_clean.X_train, split_clean.y_train)
    results["random_forest_clean"] = rf_clean.evaluate(split_clean.X_val, split_clean.y_val)

    rf_eng = RandomForestModel("RF_engineered")
    rf_eng.train(split_eng.X_train, split_eng.y_train)
    results["random_forest_engineered"] = rf_eng.evaluate(split_eng.X_val, split_eng.y_val)

    # Save engineered training data
    eng_train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)

    # Predict for submission
    if PREDICT_SUBMISSION:
        preds = rf_eng.predict(eng_test_df.drop(columns=["PassengerId"]))
        submission = eng_test_df[["PassengerId"]].copy()
        submission[TARGET_COLUMN] = preds
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"✅ Saved predictions to {SUBMISSION_PATH}")

    # Print summary
    print("\n=== Summary ===")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.4f}")


if __name__ == "__main__":
    run()
