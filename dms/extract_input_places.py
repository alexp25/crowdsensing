
import pandas as pd
from modules import loader

content = loader.read_csv2("coords_nearby_filtered.csv", 'google_id')

print(content)

place_ids = []
place_ids = content["google_id"]
place_ids = ["place_id:" + pid for pid in place_ids]
print(place_ids)

