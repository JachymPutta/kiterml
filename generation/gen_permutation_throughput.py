from constants import DATA_LOCATION, OUT_DATA_FILE

data = []

with open(DATA_LOCATION) as data_file:
    for row in data_file:
        num_list = list(map(int, row.split(' ')))
        data.append(num_list)


def update_dictionary(dictionary, key, value):
    if value not in dictionary.get(key, []):
        dictionary.setdefault(key, []).append(value)

all_permutations = {}

for sublist in data:
    res = sublist[-1:][0]
    sublist = sublist[:-1]
    key = ','.join(map(str, sublist))
    update_dictionary(all_permutations, key, res)


with open(OUT_DATA_FILE, 'w') as out_data_file:
    for k, v in all_permutations.items():
        line = f"{k}: {v}\n"
        out_data_file.write(line)


        
