from collections import Counter
import numpy as np
import pandas as pd

def Reweighing(X, Y, A):
    # Use X[A] to get the columns, then ensure it's treated as a Series/1D for the dict
    # If A is a list with one element, .squeeze() converts the DataFrame to a Series
    sensitive_col = X[A].squeeze()
    
    df = pd.DataFrame({'a': sensitive_col, 'y': Y})
    n = len(df)
    
    # ... rest of the code stays the same ...
    sample_weight = np.zeros(n)
    
    p_y = df['y'].value_counts(normalize=True).to_dict()
    p_a = df['a'].value_counts(normalize=True).to_dict()
    p_ay = (df.groupby(['a', 'y']).size() / n).to_dict()
    
    for (val_a, val_y), prob_ay in p_ay.items():
        weight = (p_y[val_y] * p_a[val_a]) / prob_ay
        mask = (df['a'] == val_a) & (df['y'] == val_y)
        sample_weight[mask] = weight

    sample_weight = sample_weight * len(Y) / np.sum(sample_weight)
    return sample_weight


