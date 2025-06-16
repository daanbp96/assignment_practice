from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

def make_predictions(
    model: RandomForestClassifier,
    test_df: DataFrame
) -> DataFrame:
    """
    Predicts the target on test data and prepares submission DataFrame.
    
    Args:
        model: Trained RandomForestClassifier.
        test_df: Test features DataFrame.
        
    Returns:
        DataFrame with 'PassengerId' and 'Survived' columns ready for submission.
    """
    preds = model.predict(test_df)
    submission = test_df[['PassengerId']].copy()
    submission['Survived'] = preds
    return submission
