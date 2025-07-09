import pandas as pd

from src.clean import clean_data
from src.features import enrich_rf
from src.models.random_forest import RandomForestModel
from src.models.utils import split_train_test

# === Constants ===
TARGET_COLUMN = "Survived"
PREDICT_SUBMISSION = True
DEV_MODE = True

TRAIN_PATH = "RandomForest/Titanic/data/raw/train.csv"
TEST_PATH = "RandomForest/Titanic/data/raw/test.csv"
CLEANED_TRAIN_PATH = "RandomForest/Titanic/data/cleaned/train.csv"
CURATED_TRAIN_PATH = "RandomForest/Titanic/data/curated/train.csv"
SUBMISSION_PATH = "RandomForest/Titanic/data/results/submission_random_forest_engineered.csv"

def run():
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Clean and store data
    cleaned_train_df = clean_data(train_df)
    cleaned_test_df = clean_data(test_df)
    cleaned_train_df.to_csv(CLEANED_TRAIN_PATH, index=False)

    # Feature engineering
    cur_train_df = enrich_rf(cleaned_train_df)
    cur_test_df = enrich_rf(cleaned_test_df)
    cur_train_df.to_csv(CURATED_TRAIN_PATH, index=False)


    # Split cleaned and engineered
    split_clean = split_train_test(cleaned_train_df, TARGET_COLUMN)
    split_eng = split_train_test(cur_train_df, TARGET_COLUMN)

    # Train and evaluate models
    results = {}

    rf_clean = RandomForestModel("RF_clean")
    rf_clean.train(split_clean.X_train, split_clean.y_train)
    results["random_forest_clean"] = rf_clean.evaluate(split_clean.X_val, split_clean.y_val)

    rf_eng = RandomForestModel("RF_engineered")
    rf_eng.train(split_eng.X_train, split_eng.y_train)
    results["random_forest_engineered"] = rf_eng.evaluate(split_eng.X_val, split_eng.y_val)

    # Predict for submission
    if PREDICT_SUBMISSION:
        preds = rf_eng.predict(cur_train_df.drop(columns=["PassengerId"]))
        submission = cur_train_df[["PassengerId"]].copy()
        submission[TARGET_COLUMN] = preds
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"✅ Saved predictions to {SUBMISSION_PATH}")

    # Print summary
    if DEV_MODE:
        rf_clean.cross_val_score(split_clean.X_train, split_clean.y_train)
        rf_eng.cross_val_score(split_eng.X_train, split_eng.y_train)
        print("\n=== Summary ===")
        for model_name, acc in results.items():
            print(f"{model_name}: {acc:.4f}")

if __name__ == "__main__":
    run()