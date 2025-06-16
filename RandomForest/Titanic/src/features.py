import pandas as pd

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    # Family size: sum of siblings/spouses, parents/children, plus the passenger themself
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Encode Sex to numeric (0/1)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # One-hot encode Embarked (3 categories)
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)

    # Drop original columns that are no longer needed
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'])

    return df
