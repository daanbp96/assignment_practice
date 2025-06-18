import pandas as pd

def add_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a categorical age bin feature to the dataframe.
    Bins example:
    - Child: 0-12
    - Teen: 13-19
    - Adult: 20-59
    - Senior: 60+

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with an 'Age' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with a new 'AgeBin' categorical column.
    """
    bins = [0, 12, 19, 59, 120]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['AgeBin'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    
    # Optional: fill missing ages with a new category 'Unknown'
    df['AgeBin'] = df['AgeBin'].cat.add_categories('Unknown').fillna('Unknown')

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    # Family size: sum of siblings/spouses, parents/children, plus the passenger themself
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Encode Sex to numeric (0/1)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # One-hot encode Embarked (3 categories)
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)

    # Add Age bins (categorical)
    df = add_age_bins(df)

    # One-hot encode AgeBin to numeric columns
    agebin_dummies = pd.get_dummies(df['AgeBin'], prefix='AgeBin')
    df = pd.concat([df, agebin_dummies], axis=1)

    # Drop original columns that are no longer needed
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked', 'AgeBin'])

    return df

