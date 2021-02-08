
import pandas as pd


def read_csv(filename):
    content = []
    with open("coords_nearby.csv", "r", encoding="utf-8") as f:
        content = []
        c = f.read().split("\n")
        for c1 in c:
            content.append(c1.split(","))
    return content


def read_csv2(filename):
    df = pd.read_csv('coords_nearby.csv')
    df = df.drop_duplicates(subset=['google_id'])
    return df


def write_csv(filename, content):
    content.to_csv(filename, index=False)


content = read_csv2("coords_nearby.csv")

print(content)

write_csv("coords_nearby_filtered.csv", content)
