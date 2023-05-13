from constants import DATA_LOCATION, OUT_DATA_FILE, GRAPH_SIZE

data = []

with open(DATA_LOCATION) as data_file:
    for row in data_file:
        num_list = list(map(int, row.split(' ')))
        data.append(num_list)


def get_cycle_permutations(key):
    if GRAPH_SIZE == 2:
        return [[key[0], key[1]], [key[1], key[0]]]
    elif GRAPH_SIZE == 4:
        l1 = [key[0], key[1], key[2], key[3]]
        l2 = [key[1], key[2], key[3], key[0]]
        l3 = [key[2], key[3], key[0], key[1]]
        l4 = [key[3], key[0], key[1], key[2]]
        return [l1, l2, l3, l4]
    else:
        raise Exception("min_max_throughput: unsupported graph size")

def update_dictionary(dictionary, key, value):
    all_keys = get_cycle_permutations(key)
    inserted = False
    for key in all_keys:
        str_key = ','.join(map(str, key))
        if str_key in dictionary.keys():
            inserted = True
            if value not in dictionary.get(str_key, []):
                dictionary.setdefault(str_key, []).append(value)
        if inserted:
            break

    if not inserted:
        str_key = ','.join(map(str, key))
        if value not in dictionary.get(str_key, []):
            dictionary.setdefault(str_key, []).append(value)


all_permutations = {}
for sublist in data:
    res = sublist[-1:][0]
    sublist = sublist[:-1]
    update_dictionary(all_permutations, sublist, res)

with open(OUT_DATA_FILE, 'w') as out_data_file:
    for k, v in all_permutations.items():
        line = f"{k}: {v}\n"
        out_data_file.write(line)


        
