from collections import Counter
import numpy as np
import pandas as pd

def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)
    # Write your code below:
    df = pd.DataFrame({'a': X[A], 'y': Y})
    n = len(df)
    
    sample_weight = np.zeros(n)
    
    p_y = df['y'].value_counts(normalize=True).to_dict()
    p_a = df['a'].value_counts(normalize=True).to_dict()
    p_ay = (df.groupby(['a', 'y']).size() / n).to_dict()
    
    for (val_a, val_y), prob_ay in p_ay.items():
        weight = (p_y[val_y] * p_a[val_a]) / prob_ay
        mask = (df['a'] == val_a) & (df['y'] == val_y)
        sample_weight[mask] = weight

    # Rescale the sum of sample weights to len(y) before returning it
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight


