import pandas as pd
from src.clean import clean_data
from src.features import generate_features
from src.predict import make_predictions
from src.models.random_forest import train_model, evaluate_model

def main():
    model_name = "random_forest_with_features"

    # === 1. Load Data ===
    train = pd.read_csv("RandomForest/Titanic/data/raw/train.csv")
    test = pd.read_csv("RandomForest/Titanic/data/raw/test.csv")

    # === 2. Clean ===
    train = clean_data(train)
    test = clean_data(test)

    # === 3. Feature Engineering ===
    train = generate_features(train)
    test = generate_features(test)

    train.to_csv("RandomForest/Titanic/data/processed/train.csv", index=False)

    # === 4. Train ===
    model, X_val, y_val = train_model(train)

    # === 5. Evaluate ===
    acc = evaluate_model(model, X_val, y_val, model_name=model_name)

    # === 6. Predict on Test ===
    submission = make_predictions(model, test)
    submission_path = f"RandomForest/Titanic/data/results/submission_{model_name}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✅ Saved predictions to {submission_path}")

if __name__ == "__main__":
    main()
