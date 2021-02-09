
import pandas as pd
from modules import loader

content = loader.read_csv2("coords_nearby.csv")
print(content)
loader.write_csv("coords_nearby_filtered.csv", content)
