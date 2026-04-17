import pandas as pd

def load_data():
    df = pd.read_csv("heart.csv")
    df.dropna(inplace=True)
    return df