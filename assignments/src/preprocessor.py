import numpy as np
import pandas as pd

def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame or dict)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)

    n = len(Y)

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Normalize A to always be a list
    if isinstance(A, str):
        A = [A]
    else:
        A = list(A)

    # Combine multiple sensitive attributes into a single composite tuple key
    a_values = X[A].apply(tuple, axis=1) if len(A) > 1 else X[A[0]]

    df = pd.DataFrame({'a': a_values, 'y': Y})

    p_y  = df['y'].value_counts(normalize=True).to_dict()
    p_a  = df['a'].value_counts(normalize=True).to_dict()
    p_ay = (df.groupby(['a', 'y']).size() / n).to_dict()

    sample_weight = np.zeros(n)

    for (val_a, val_y), prob_ay in p_ay.items():
        weight = (p_y[val_y] * p_a[val_a]) / prob_ay
        mask = (df['a'] == val_a) & (df['y'] == val_y)
        sample_weight[mask] = weight

    # Rescale so weights sum to len(Y)
    sample_weight = sample_weight * n / sample_weight.sum()

    return sample_weight


