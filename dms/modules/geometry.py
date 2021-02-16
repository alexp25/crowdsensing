import math

# generate random Gaussian values
from random import seed
from random import gauss
import random
import time


def init_random(fixed):
    # seed random number generator
    if fixed:
        seed(1)
    else:
        seed(int(time.time()))


def get_distance_from_deg(c1, c2):
    # approximate radius of earth in m
    R = 6373.0 * 1000

    lat1 = math.radians(c1[0])
    lon1 = math.radians(c1[1])
    lat2 = math.radians(c2[0])
    lon2 = math.radians(c2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    # print("Result:", distance)
    return distance


def get_center_point(points):
    # https://stackoverflow.com/questions/37885798/how-to-calculate-the-midpoint-of-several-geolocations-in-python
    x = 0.0
    y = 0.0
    z = 0.0

    for coord in points:
        lat = math.radians(coord[0])
        lng = math.radians(coord[1])

        x += math.cos(lat) * math.cos(lng)
        y += math.cos(lat) * math.sin(lng)
        z += math.sin(lat)

    total = len(points)

    x = x / total
    y = y / total
    z = z / total

    central_lng = math.atan2(y, x)
    central_square_root = math.sqrt(x * x + y * y)
    central_lat = math.atan2(z, central_square_root)

    return [math.degrees(central_lat), math.degrees(central_lng)]


def get_random_point_in_radius(c, min_radius, max_radius):
    min_radius /= 1000
    max_radius /= 1000
    t = 2 * math.pi * random.random()
    u = random.random() + random.random()
    r = math.radians(max_radius - min_radius) * \
        (2 - u if u > 1 else u) + math.radians(min_radius)

    deltax = r * math.cos(t)
    deltay = r * math.sin(t)

    return [c[0] + deltax, c[1] + deltay]


def get_point_on_heading(c, distance, heading):
    t = heading * (math.pi/180)
    r = (distance / 1000) * (math.pi / 360)

    deltax = r * math.cos(t)
    deltay = r * math.sin(t)

    print(deltax, deltay)

    return [c[0] + deltax, c[1] + deltay]

