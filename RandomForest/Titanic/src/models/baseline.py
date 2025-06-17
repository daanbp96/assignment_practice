from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_baseline_model(df, target_column="Survived", strategy="most_frequent"):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DummyClassifier(strategy=strategy, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"[BaselineModel] Accuracy (strategy={strategy}): {acc:.4f}")
    return model, acc
