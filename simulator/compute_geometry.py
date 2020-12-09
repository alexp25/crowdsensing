import yaml
import csv
from modules import geometry


def load_config():
    """load global config"""

    global config, input_spec, coords
    with open('config.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        input_file = config["coords_file"]

    with open(config["input_file"]) as file:
        input_spec = yaml.load(file, Loader=yaml.FullLoader)

    with open(input_file) as file:
        csvdata = csv.reader(file)
        coords = []
        nrows = 0
        for row in csvdata:
            r = [float(e) for e in row]
            ncols = len(r)
            coords.append(r)
            nrows += 1

        print("msize: ", nrows, ncols)
        print(coords)


def compute_distance_matrix(coords):
    """build distance matrix from coords"""

    d_mat = []
    for c1 in coords:
        # get distance between current point and all other points
        d_vect = []
        for c2 in coords:
            d = int(geometry.get_distance_from_deg(c1, c2))
            d_vect.append(d)
        d_mat.append(d_vect)

    return d_mat

def compute_distance_matrix_wrapper():
    return compute_distance_matrix(coords)

def compute_random_points(coords):
    """set origin as coords centroid. compute distances relative to origin. update depot coords with random points."""

    new_coords = []
    for c in coords:
        nc = [c[0], c[1]]
        new_coords.append(nc)

    n_vehicles = len(input_spec["depots"])

    # override vehicle coords init within bounding circle
    center_point = geometry.get_center_point(coords)

    max_dist_from_center = 0
    for c in coords:
        dist = geometry.get_distance_from_deg(c, center_point)
        if dist > max_dist_from_center:
            max_dist_from_center = dist

    for i in range(n_vehicles):
        new_coords[i] = geometry.get_random_point_in_radius(
            center_point, 0, max_dist_from_center)

    return new_coords


def main():
    """main function: compute distance matrix for given coords"""

    d_mat = compute_distance_matrix(coords)
    print(d_mat)
    with open(config["matrix_file"], "w") as file:
        for row in d_mat:
            file.write(",".join([str(e) for e in row]) + "\n")

def get_distance_matrix_with_random_depots():
    new_coords = compute_random_points(coords)
    d_mat = compute_distance_matrix(new_coords)
    return d_mat

if __name__ == "__main__":
    load_config()
    # main()
    geometry.init_random(False)
    d_mat = compute_distance_matrix(coords)
    print(d_mat)
    d_mat = get_distance_matrix_with_random_depots()
    print(d_mat)
 
