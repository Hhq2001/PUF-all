import pandas as pd
import numpy as np

def process_dielectric_loss(df):
    values = df.iloc[:50, 2].astype(float)
    return values.values.reshape(-1, 10).T

def generate_bitstream(values, indices):
    reordered = [values[i - 1] for i in indices if i - 1 < len(values)]
    median = np.median(reordered)

    return [1 if v > median else 0 for v in reordered]