import json


with open("data/routes_sim_vcg_multi.txt", "r") as f:
    data = f.read()
    data = json.loads(data)


adata_vect = []

for run in data:
    adata = []
    for v in run:
        # "vehicle": 0, "points": 2, "distance": 1174, "load": 9
        adata.append({
            "vehicle": v["player_id"],
            "points": v["issues"],
            "distance": 0,
            "load": v["total"]
        })

    adata_vect.append(adata)

with open("data/routes_sim_vcg_multi.adapted.txt", "w") as f:
    f.write(json.dumps(adata_vect))
