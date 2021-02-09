
import requests
import json
import urllib
import yaml
import config_loader
import time

from modules import geometry


def create_distance_matrix():
    pagetoken = None
    places_vect = []
    n_pages_max = 10
    n_pages_max = 5

    full_radius = 5000

    radius = 1000
    # radius = None
    loctype = "restaurant"
    # loctype = None

    lat = 44.4566428
    lng = 26.0808892

    c = [lat, lng]
    c_vect = []
    pivot = c

    n_cells = int(full_radius / radius)

    for i in range(n_cells):
        c_row = []
        c = pivot
        for j in range(n_cells):
            c = geometry.get_point_on_heading(c, radius, 0)
            c_row.append(c)
        c_vect.append(c_row)
        pivot = geometry.get_point_on_heading(pivot, radius, 90)
    
    # for row in c_vect:
    #     print(row)   

    for row in range(n_cells):
        for col in range(n_cells):
            for i in range(n_pages_max):
                center = c_vect[row][col]
                print("row: ", row, "col: ", col,
                      "request: ", i+1, "/", n_pages_max)
                places_vect_k, pagetoken = request_coords(
                    pagetoken, center[0], center[1], radius, loctype)
                places_vect.extend(places_vect_k)
                print("complete")
                if pagetoken is None:
                    print("no more pages")
                    time.sleep(1)
                    break
                time.sleep(1)

    return places_vect


def request_coords(pagetoken, lat, lng, radius, loctype):
    API_key = 'AIzaSyCIESqE0Ghd54R0qOtvAQ2WmGI2M3_TbX4'
    request = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?'
    request += "&key=" + API_key
    request += "&location="+str(lat)+","+str(lng)
    # 44.4531131
    # 26.0846382
    if radius is not None:
        request += "&radius=" + str(radius)
    if loctype is not None:
        request += "&type=" + loctype
    if pagetoken is not None:
        request += "&pagetoken=" + pagetoken

    # print(request)

    jsonResult = urllib.request.urlopen(request).read()
    response = json.loads(jsonResult)

    # print(response)
    if "next_page_token" in response:
        pagetoken = response["next_page_token"]
    else:
        pagetoken = None
    places = response['results']
    places_vect = []

    for place in places:
        loc = place["geometry"]["location"]
        place_extra = {
            "google_id": place["place_id"],
            "name": place["name"],
            "lat": str(loc['lat']),
            "lng": str(loc["lng"])
        }
        places_vect.append(place_extra)

    return places_vect, pagetoken


def main():
    """Entry point of the program"""
    places_vect = create_distance_matrix()

    with open("coords_nearby.csv", "w", encoding="utf-8") as f:
        for i, place in enumerate(places_vect):
            row = []
            keys = []
            for k in place:
                row.append(place[k])
                keys.append(k)

            if i == 0:
                row_str = ",".join(keys)
                f.write(row_str + "\n")

            row_str = ",".join(row)
            f.write(row_str + "\n")


if __name__ == '__main__':
    main()
