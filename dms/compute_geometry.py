import yaml
import csv
from modules import geometry, plotter
from modules.random_walk import RandomWalk
import config_loader
import numpy as np

config, input_spec, coords, dm = config_loader.load_config()
random_walks = []
random_walk_record = []


def set_coords(new_coords):
    global coords
    coords = new_coords


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


def compute_random_points(coords, random_walk):
    """set origin as coords centroid. compute distances relative to origin. update depot coords with random points."""

    new_coords = []
    for c in coords:
        nc = [c[0], c[1]]
        new_coords.append(nc)

    if input_spec["options"]["fixed_vehicles"]:
        n_vehicles = input_spec["options"]["n_vehicles"]
    else:
        n_vehicles = len(input_spec["depots"])

    # override vehicle coords init within bounding circle (excl. vehicles)
    center_point = geometry.get_center_point(coords[n_vehicles:])

    max_dist_from_center = 0
    for c in coords[n_vehicles:]:
        dist = geometry.get_distance_from_deg(c, center_point)
        if dist > max_dist_from_center:
            max_dist_from_center = dist

    for i in range(n_vehicles):
        if random_walk:
            new_coords[i] = random_walks[i].bound_walk_step(
                input_spec["options"]["random_walk_step"])
            # record walk
            random_walk_record[i].append(new_coords[i])
        else:
            new_coords[i] = geometry.get_random_point_in_radius(
                center_point, 0, max_dist_from_center)

    if random_walk:
        # incremental update coords
        coords[0:n_vehicles] = new_coords[0:n_vehicles]
        
    return new_coords


def main():
    """main function: compute distance matrix for given coords"""
    d_mat = compute_distance_matrix(coords)
    print(d_mat)
    with open(config["matrix_file"], "w") as file:
        for row in d_mat:
            file.write(",".join([str(e) for e in row]) + "\n")


def get_distance_matrix_with_random_depots():
    new_coords = compute_random_points(coords, False)
    d_mat = compute_distance_matrix(new_coords)
    return d_mat


def init_random_walk(vehicles, radius):
    global random_walks, random_walk_record

    print("init random walks")
    center_point = geometry.get_center_point(coords)

    max_dist_from_center = 0
    if radius is not None:
        max_dist_from_center = radius
    else:
        for c in coords:
            dist = geometry.get_distance_from_deg(c, center_point)
            if dist > max_dist_from_center:
                max_dist_from_center = dist

    random_walks = []
    random_walk_record = []

    for v in vehicles:
        random_walks.append(RandomWalk(center_point, max_dist_from_center + input_spec["options"]["extra_coverage_dist"]))
        random_walk_record.append([])


def get_distance_matrix_with_random_walk():
    new_coords = compute_random_points(coords, True)
    d_mat = compute_distance_matrix(new_coords)
    return d_mat


def plot_random_walk_record():
    random_walk_record_transposed = random_walk_record
    return plotter.plot_coords_map_multi(random_walk_record_transposed)


if __name__ == "__main__":
    # main()
    geometry.init_random(False)
    d_mat = compute_distance_matrix(coords)
    print(d_mat)
    d_mat = get_distance_matrix_with_random_depots()
    print(d_mat)
