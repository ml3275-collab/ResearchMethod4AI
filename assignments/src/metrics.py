import numpy as np

class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = np.array(y)
        self.y_pred = np.array(y_pred)

    def acc(self):
        # Accuracy
        return 1.0 - np.sum(np.abs(self.y - self.y_pred)) / len(self.y)

    def eod(self, s):
        # Equal Opportunity Difference
        s = np.array(s)
        mask1, mask0 = s == 1, s == 0

        tpr1 = np.sum((self.y[mask1] == 1) & (self.y_pred[mask1] == 1)) / np.sum(self.y[mask1] == 1)
        tpr0 = np.sum((self.y[mask0] == 1) & (self.y_pred[mask0] == 1)) / np.sum(self.y[mask0] == 1)

        return tpr1 - tpr0

    def aod(self, s):
        # Average Odds Difference
        s = np.array(s)
        mask1, mask0 = s == 1, s == 0

        tpr1 = np.sum((self.y[mask1] == 1) & (self.y_pred[mask1] == 1)) / np.sum(self.y[mask1] == 1)
        tpr0 = np.sum((self.y[mask0] == 1) & (self.y_pred[mask0] == 1)) / np.sum(self.y[mask0] == 1)

        fpr1 = np.sum((self.y[mask1] == 0) & (self.y_pred[mask1] == 1)) / np.sum(self.y[mask1] == 0)
        fpr0 = np.sum((self.y[mask0] == 0) & (self.y_pred[mask0] == 1)) / np.sum(self.y[mask0] == 0)

        return (tpr1 - tpr0 + fpr1 - fpr0) / 2.0

    def spd(self, s):
        # Statistical Parity Difference
        s = np.array(s)
        mask1, mask0 = s == 1, s == 0

        pr1 = np.sum(self.y_pred[mask1] == 1) / np.sum(mask1)
        pr0 = np.sum(self.y_pred[mask0] == 1) / np.sum(mask0)

        return np.abs(pr1 - pr0)