"""
TODO: Enter any documentation that only people updating the metric should read here.

All columns of the solution and submission dataframes are passed to your metric, except for the Usage column.

Your metric must satisfy the following constraints:
- You must have a function named score. Kaggle's evaluation system will call that function.
- You can add your own arguments to score, but you cannot change the first three (solution, submission, and row_id_column_name).
- All arguments for score must have type annotations.
- score must return a single, finite, non-null float.
"""

import pandas as pd
import pandas.api.types
import numpy as np


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''

    https://docs.python.org/3/library/doctest.html
    # This example doctest works for mean absolute error:
    >>> import pandas as pd
    >>> y_pred = pd.DataFrame({"pred": [0, 2, 1, 3]})
    >>> y_pred["id"] = range(len(y_pred))
    >>> y_true = pd.DataFrame({"pred": [0, 1, 2, 3]})
    >>> y_true["id"] = range(len(y_true))
    >>> score(y_true.copy(), y_pred.copy(), "id")
    0.37796447300922725
    '''

    return np.sqrt(np.sum((solution["pred"] - submission["pred"]) ** 2)) / np.sqrt(np.sum(solution["pred"] ** 2))
