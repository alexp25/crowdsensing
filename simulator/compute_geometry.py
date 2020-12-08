import yaml
import csv
from modules import geometry

with open('config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    input_file = config["coords_file"]

with open(config["input_file"]) as file:
    input_spec = yaml.load(file, Loader=yaml.FullLoader)

with open(input_file) as file:
    csvdata = csv.reader(file)
    dm = []
    nrows = 0

    for row in csvdata:
        r = [float(e) for e in row]
        ncols = len(r)
        dm.append(r)
        nrows += 1

    print("msize: ", nrows, ncols)
    print(dm)

    # set origin as dm[0]
    # compute distances relative to origin
    # build distance matrix

    d_mat = []

    n_vehicles = len(input_spec["depots"])

    print(n_vehicles)

    # override vehicle coords init within bounding circle

    center_point = geometry.get_center_point(dm)
    print(center_point)

    max_dist_from_center = 0
    for c in dm:
        dist = geometry.get_distance_from_deg(c, center_point)
        if dist > max_dist_from_center:
            max_dist_from_center = dist

    print(max_dist_from_center)

    geometry.init_random(False)

    for i in range(10):
        print(geometry.get_random_point_in_radius(center_point, 0, max_dist_from_center))
    quit()

    for c1 in dm:
        # get distance between current point and all other points
        d_vect = []
        for c2 in dm:
            d = int(geometry.get_distance_from_deg(c1, c2))
            d_vect.append(d)
        d_mat.append(d_vect)

    print(d_mat)

with open(config["matrix_file"], "w") as file:
    for row in d_mat:
        file.write(",".join([str(e) for e in row]) + "\n")
