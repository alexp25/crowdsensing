from __future__ import division
from __future__ import print_function
import requests
import json
import urllib
import yaml

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
    # Distance Matrix API only accepts 100 elements per request, so get rows in multiple requests.
    max_elements = 100
    num_addresses = len(addresses)  # 16 in this example.
    # Maximum number of rows that can be computed per request (6 in this example).
    max_rows = max_elements // num_addresses
    # num_addresses = q * max_rows + r (q = 2 and r = 4 in this example).
    q, r = divmod(num_addresses, max_rows)
    dest_addresses = addresses
    distance_matrix = []
    # Send q requests, returning max_rows rows per request.
    for i in range(q):
        origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
        response = send_request(origin_addresses, dest_addresses, API_key)
        distance_matrix += build_distance_matrix(response)

    # Get the remaining remaining r rows, if necessary.
    if r > 0:
        origin_addresses = addresses[q * max_rows: q * max_rows + r]
        response = send_request(origin_addresses, dest_addresses, API_key)
        distance_matrix += build_distance_matrix(response)
    return distance_matrix


def send_request(origin_addresses, dest_addresses, API_key):
    """ Build and send request for the given origin and destination addresses."""
    def build_address_str(addresses):
        # Build a pipe-separated string of addresses
        address_str = ''
        for i in range(len(addresses) - 1):
            address_str += addresses[i] + '|'
        address_str += addresses[-1]
        return address_str

    request = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=metric'
    # origin_addresses = [origin_addresses[0], origin_addresses[1]]
    origin_address_str = build_address_str(origin_addresses)
    # dest_addresses = [dest_addresses[0], dest_addresses[1]]
    dest_address_str = build_address_str(dest_addresses)
    request = request + '&origins=' + origin_address_str + '&destinations=' + \
        dest_address_str + '&key=' + API_key
    print("origin (" + str(len(origin_addresses)) + ")")
    print(origin_address_str)
    print("dest (" + str(len(dest_addresses)) + ")")
    print(dest_address_str)
    print("request")
    print(request)
    print("\n")
    # quit()
    jsonResult = urllib.request.urlopen(request).read()
    response = json.loads(jsonResult)
    return response


def build_distance_matrix(response):
    distance_matrix = []
    for row in response['rows']:
        row_list = [row['elements'][j]['distance']['value']
                    for j in range(len(row['elements']))]
        distance_matrix.append(row_list)
    return distance_matrix

########
# Main #
########


def main():
    """Entry point of the program"""

    input_file = None
    config = None

    with open('config.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        input_file = config["input_file"]

    with open(input_file) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)
        print(config)

    # Create the data.
    data = create_data(config)
    addresses = data['addresses']
    API_key = data['API_key']
    distance_matrix = create_distance_matrix(data)
    print(distance_matrix)
    with open(config["matrix_file"], "w") as f:
        for row in distance_matrix:
            f.write(",".join([str(e) for e in row]) + "\n")


if __name__ == '__main__':
    main()
