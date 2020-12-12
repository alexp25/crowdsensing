import config_loader
from modules import encode

config, input_data, coords, dm = config_loader.load_config()


def extract_custom_dm(dm1):
    return extract(config, input_data, coords, dm1)

def extract(config, input_data, coords, dm):

    depots = input_data["depots"]
    issues = input_data["issues"]
    vehicles = input_data["vehicles"]

    # print(depots)
    # print(issues)
    # print(vehicles)

    issues = [str(i + 1) for i in range(len(issues))]

    items_to_sell = issues
    bidders = vehicles

    bids_vect = []
    for i in range(len(vehicles)):
        bids = []
        for j in range(len(depots), len(depots) + len(issues)):
            bids.append({
                "item": issues[j-len(depots)],
                "bid": int(dm[i][j] * (vehicles[i]["capacity"]/100))
            })
        bids_vect.append(bids)

    output_items_to_sell = "ITEMS TO SELL:\n"
    output_items_to_sell += "\n".join(["issue_" + str(item)
                                       for item in items_to_sell]) + "\n----\n"
    output_name_bids = ""

    for i, bids in enumerate(bids_vect):
        v = vehicles[i]
        output_name_bids += "Name: " + encode.encode_vehicle_name(v) + "\n"
        output_name_bids += "Bids:\n"
        for bid in bids:
            output_name_bids += "\t" + \
                str(bid["bid"]) + "\tissue_" + str(bid["item"]) + "\n"
        output_name_bids += "-----\n"

    output_vcg = output_items_to_sell + output_name_bids
    return output_vcg


def main():
    output_vcg = extract(config, input_data, coords, dm)
    print(output_vcg)

    with open("input_vcg.txt", "w") as f:
        f.write(output_vcg)


if __name__ == "__main__":
    main()
