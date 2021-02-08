

from __future__ import division
from __future__ import print_function
import requests
import json
import urllib
import yaml
import config_loader

# https://developers.google.com/maps/documentation/distance-matrix/overview

def create_data(config):
    """Creates the data."""
    data = {}
    data['API_key'] = 'AIzaSyCIESqE0Ghd54R0qOtvAQ2WmGI2M3_TbX4'
    
    depots = config["depots"]
    issues = config["issues"]

    data['addresses'] = depots + issues
    return data


def create_distance_matrix(data):
    addresses = data["addresses"]
    API_key = data["API_key"]
    coords_vect = []
    for place in addresses:
        coords = request_coords(place, API_key)
        coords_vect.append(coords)

    return coords_vect


def request_coords(place_id, API_key):
    place_id = place_id[9:]
    request = 'https://maps.googleapis.com/maps/api/geocode/json?place_id=' + place_id

    request += "&key=" + API_key

    print(request)

    # quit()
    jsonResult = urllib.request.urlopen(request).read()
    response = json.loads(jsonResult)

    # print(response)
    loc = response['results'][0]['geometry']['location']
    coords = [loc['lat'], loc['lng']]
    return coords


########
# Main #
########


def main():
    """Entry point of the program"""

    input_file = None
    config = None

    config, input_data, coords, dm = config_loader.load_config()

    # Create the data.
    data = create_data(input_data)
    coords_vect = create_distance_matrix(data)
    # coords_vect = [[44.4352806, 26.0490164], [44.448984, 26.0561547], [44.4421364, 26.0666876], [44.4386435, 26.0458386], [44.4430103, 26.05114], [44.4447228, 26.0548487], [44.4482361, 26.0429976], [44.4492548, 26.0530481], [44.4384469, 26.038602], [44.4430707, 26.0627016], [44.4468016, 26.0660459], [44.4404178, 26.0629714], [44.4335841, 26.0560158]]
    print(coords_vect)
    with open(config["coords_file"], "w") as f:
        for row in coords_vect:
            f.write(",".join([str(e) for e in row]) + "\n")


if __name__ == '__main__':
    main()

