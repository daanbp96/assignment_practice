import pandas as pd
import numpy as np
from word2number import w2n
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# === STEP 1: Load data ===
# (Assumes you have `data` and `test_data` DataFrames already loaded)

# === STEP 2: Convert worded numbers ===
def word_to_num(value):
    try:
        if isinstance(value, str):
            return w2n.word_to_num(value.replace(",", ""))
        return value
    except:
        return np.nan

for col in ['Height(px)', 'Width(px)', 'InternalStorage(MB or GB)']:
    data[col] = data[col].apply(word_to_num)
    test_data[col] = test_data[col].apply(word_to_num)

# === STEP 3: Normalize Units ===
def normalize_ram(val):
    try:
        return val / 1024 if val > 2048 else val
    except:
        return np.nan

def normalize_storage(val):
    try:
        return val * 1024 if val < 1000 else val
    except:
        return np.nan

def normalize_weight(val):
    try:
        return val * 1000 if val < 10 else val
    except:
        return np.nan

for df in [data, test_data]:
    df['RAM'] = df['RAM'].apply(normalize_ram)
    df['InternalStorage(MB or GB)'] = df['InternalStorage(MB or GB)'].apply(normalize_storage)
    df['Weight(g or Kg)'] = df['Weight(g or Kg)'].apply(normalize_weight)

    df.rename(columns={
        'InternalStorage(MB or GB)': 'InternalStorage_MB',
        'Weight(g or Kg)': 'Weight_g'
    }, inplace=True)

# === STEP 4: Remove Outliers using IQR ===
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Drop NaNs created during conversion
data.dropna(inplace=True)

# Remove outliers from numeric columns (excluding ID and target)
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.drop(['p_id', 'target'])
data = remove_outliers_iqr(data, numeric_cols)

# === STEP 5: Train Model ===
X = data.drop(columns=['p_id', 'target'])
y = data['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Validation F1 Score:", f1_score(y_val, y_pred))

# === STEP 6: Predict and Submit ===
X_test = test_data[X.columns]
test_preds = model.predict(X_test)

submission = pd.DataFrame({
    'p_id': test_data['p_id'],
    'target': test_preds
})

submission.to_csv('submission.csv', index=False)
print("✅ submission.csv saved")
