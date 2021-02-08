import yaml
import csv

def load_config():
    """load global config"""

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

        # print("msize: ", nrows, ncols)
        # print(coords)
    
    with open(config["matrix_file"]) as file:
        csvdata = csv.reader(file)
        dm = []
        nrows = 0

        for row in csvdata:
            r = [int(e) for e in row]
            ncols = len(r)
            dm.append(r)
            nrows += 1

        # print("msize: ", nrows, ncols)

    return config, input_spec, coords, dm
