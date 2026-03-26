import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    X = df[["area", "bedrooms"]]
    y = df["price"]
    return X, y
