import os
import random

from constants import GRAPH_SIZE, RANDOM_SEED

def denormalize_graph(g):
    weights = g[:GRAPH_SIZE]
    tokens = g[GRAPH_SIZE:]
    deltas = random.sample(range(1,10), 4)

    res_tokens = [n1 * n2 for n1,n2 in zip(tokens, deltas)]
    res_weights = [weights[0] * deltas[3], weights[0] * deltas[0],
                   weights[1] * deltas[0], weights[1] * deltas[1],
                   weights[2] * deltas[1], weights[2] * deltas[2],
                   weights[3] * deltas[2], weights[3] * deltas[0]]

    # return res_weights + res_tokens
    #
    return res_weights + [res_tokens[3]]


def denormalize(data):
    return [denormalize_graph(gr) for gr in data]

def gen_denormalized(data_loc, out_loc):
    data = []
    random.seed(RANDOM_SEED)

    print(os.getcwd())
    with open(data_loc) as data_file:
        for row in data_file:
            num_list = list(map(int, row.split(',')))
            data.append(num_list)


    denormalized_data = denormalize(data)

    with open(out_loc, 'w') as out_data_file:
        for line in denormalized_data:
            out_data_file.write(','.join(list(map(str,line))))
            out_data_file.write('\n')



