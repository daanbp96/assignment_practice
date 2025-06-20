import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning to ensure the data is model-compatible.
    - Handle missing values
    - Convert categorical to numeric (basic)
    - Drop irrelevant or high-cardinality columns
    """

    # Fill missing numeric values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Fill missing categorical values
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex'] = df['Sex'].fillna('missing')  # just in case

    # Convert Sex to numeric (0/1)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # One-hot encode Embarked
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)

    # Create HasCabin feature
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Drop irrelevant or text-heavy columns
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'])

    return df
