
import subprocess
import json

import config_loader
from modules import encode
import extract_vcg_input
import compute_geometry


def run_sim(filename = "input_vcg.txt"):
    proc = subprocess.Popen("py -3 vcg_auction.py " + filename,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    output = proc.communicate()[0]
    output = output.decode('utf-8')
    print(output)
    return output


def extract_output(output, vehicles):
    output_lines = [line for line in output.split("\r\n")]
    # print(output_lines)
    vehicle_names = [encode.encode_vehicle_name(v) for v in vehicles]
    output_data = {}
    player_crt = None
    for line in output_lines:
        next_player = False
        for vn in vehicle_names:
            if vn in line:
                next_player = True
                player_crt = vn
                output_data[vn] = {
                    "total": int(line[line.index("pays ") + 5:line.index(" for:")]),
                    "issues": 0
                }
                break
        if not next_player:
            output_data[player_crt]["issues"] += 1

    output_data_mat = []
    for k in output_data:
        d = output_data[k]
        d["player"] = k
        d["player_id"] = int(k[k.index("er_")+3:k.replace("er_", "   ").index("_")])
        output_data_mat.append(d)
    print(output_data_mat)
    return output_data_mat

def update_coords(init):
    if init:
        dm1 = compute_geometry.compute_distance_matrix_wrapper()
    else:
        dm1 = compute_geometry.get_distance_matrix_with_random_depots()
    vcg_input = extract_vcg_input.extract_custom_dm(dm1)
    print(vcg_input)
    return vcg_input

config, input_data, coords, dm = config_loader.load_config()

depots = input_data["depots"]
issues = input_data["issues"]
vehicles = input_data["vehicles"]

print(depots)
print(issues)
print(vehicles)

n_iter = 1000
extracted_vect = []

for i in range(n_iter):

    print("iteration: " + str(i+1) + "/" + str(n_iter) + " - " + str((int((i+1)/n_iter*100))) + "%" + " complete")

    if i == 0:
        vcg_input = update_coords(True)
    else:
        vcg_input = update_coords(False)

    filename = "input_vcg_temp.txt"
    with open(filename, "w") as f:
        f.write(vcg_input)

    output = run_sim(filename)
    extracted = extract_output(output, vehicles)
    extracted_vect.append(extracted)
    
with open("data/routes_sim_vcg_multi.txt", "w") as f:
    f.write(json.dumps(extracted_vect))
    