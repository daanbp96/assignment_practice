import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fill Age missing with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Fill Embarked missing with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Create HasCabin feature (boolean)
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    return df
