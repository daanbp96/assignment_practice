

import pandas as pd
from src.clean import clean_data
from src.features import generate_features
from src.model import train_model, evaluate_model
from src.predict import make_predictions

def main():
    # === 1. Load Data ===
    train = pd.read_csv("RandomForest/Titanic/data/raw/train.csv")
    test = pd.read_csv("RandomForest/Titanic/data/raw/test.csv")

    # === 2. Clean ===	
    train = clean_data(train)
    test = clean_data(test)

    # === 3. Feature Engineering ===
    train = generate_features(train)
    test = generate_features(test)

    # # === 4. Split + Train ===
    model, X_val, y_val = train_model(train)

    # # === 5. Evaluate ===
    evaluate_model(model, X_val, y_val)

    # === 6. Predict on Test ===
    submission = make_predictions(model, test)
    submission.to_csv("submission.csv", index=False)
    print("✅ Saved predictions to submission.csv")

if __name__ == "__main__":
    main()
