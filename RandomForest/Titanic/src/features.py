import pandas as pd

def add_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 12, 19, 59, 120]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['AgeBin'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    # Handle missing AgeBin
    df['AgeBin'] = df['AgeBin'].cat.add_categories('Unknown').fillna('Unknown')
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to improve model performance.
    """
    df = df.copy()

    # Add family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Add age bin categories
    df = add_age_bins(df)
    agebin_dummies = pd.get_dummies(df['AgeBin'], prefix='AgeBin')
    df = pd.concat([df, agebin_dummies], axis=1)

    # Drop intermediate columns
    df = df.drop(columns=['AgeBin'])

    return df
