import pandas as pd


def read_csv(filename):
    content = []
    with open(filename, "r", encoding="utf-8") as f:
        content = []
        c = f.read().split("\n")
        for c1 in c:
            if len(c1) > 0:
                content.append(c1.split(","))
    return content


def read_csv2(filename, duplicates_key=None):
    df = pd.read_csv(filename)
    if duplicates_key is not None:
        df = df.drop_duplicates(subset=[duplicates_key])
    return df


def write_csv(filename, content):
    content.to_csv(filename, index=False)
