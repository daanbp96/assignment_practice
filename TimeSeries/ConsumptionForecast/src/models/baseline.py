from .baseclass import ModelWrapper
import pandas as pd
from skforecast.recursive import ForecasterEquivalentDate, ForecasterRecursive



forecaster = ForecasterEquivalentDate(
                 offset    = pd.DateOffset(days=1),
                 n_offsets = 1
             )
