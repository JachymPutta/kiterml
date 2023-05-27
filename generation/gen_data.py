#!/usr/bin/env python
# coding: utf-8

import itertools
import math
import subprocess
import multiprocessing
from functools import partial


def is_coprime(numbers):
    # Check if the GCD of all the numbers is 1
    return math.gcd(*numbers) == 1


def generate_combinations(limit):
    # Generate all combinations of 4 integers within a range with possible repetition
    numbers = range(1, limit + 1)  # Adjust the range as needed
    combinations = list(itertools.product(numbers, repeat=4))

    # Generate all possible orderings of the numbers within each combination
    ordered_combinations = [list(itertools.permutations(combination)) for combination in combinations]

    # Flatten the list of orderings and filter based on the GCD condition
    coprime_combinations = [ordering for combination in ordered_combinations for ordering in combination if
                            is_coprime(ordering)]

    # Remove duplicates and sort the unique combinations in ascending order
    unique_combinations = sorted(list(set(coprime_combinations)))

    return unique_combinations


def printActors(lst, file):
    for i, el in enumerate(lst):
        file.write(f'\t\t\t<actor name="a{i + 1}" type="A">\n')
        file.write(f'\t\t\t\t<port name="in{i + 1}" type="in" rate="{el}"/>\n')
        file.write(f'\t\t\t\t<port name="out{i + 1}" type="out" rate="{el}"/>\n')
        file.write('\t\t\t</actor>\n')


def printChannels(lst, tokens, file):
    for i, el in enumerate(lst):
        if i == len(lst) - 1:
            file.write(
                f'\t\t\t<channel name="d{i + 1}" srcActor="a{i + 1}" srcPort="out{i + 1}" dstActor="a1" dstPort="in1" size="1" initialTokens="{tokens}"/>\n')
        else:
            file.write(
                f'\t\t\t<channel name="d{i + 1}" srcActor="a{i + 1}" srcPort="out{i + 1}" dstActor="a{i + 2}" dstPort="in{i + 2}" size="1" initialTokens="0"/>\n')


def printProperties(lst, file):
    for i, _ in enumerate(lst):
        file.write(f'\t\t\t<actorProperties actor="a{i + 1}">\n')
        file.write('\t\t\t\t<processor type="cluster_0" default="true">\n')
        file.write('\t\t\t\t\t<executionTime time="1"/>\n')
        file.write('\t\t\t\t</processor>\n')
        file.write('\t\t\t</actorProperties>\n')


def printGraph(lst, tokens, file):
    file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    file.write('<sdf3 version="1.0" type="sdf" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
    file.write('\t<applicationGraph name="app">\n')

    file.write('\t\t<sdf name="tmp" type="Example">\n')
    printActors(lst, file)
    printChannels(lst, tokens, file)
    file.write('\t\t</sdf>\n')

    file.write('\t\t<sdfProperties>\n')
    printProperties(lst, file)
    file.write('\t\t</sdfProperties>\n')

    file.write('\t</applicationGraph>\n')
    file.write('</sdf3>\n')


def getRes(filename, kiter_path):
    result = subprocess.run([kiter_path, '-aKPeriodicThroughput', '-f', filename], capture_output=True, text=True)
    return result.stdout


def getFlow(kiter_path, lst):
    lo, hi, mid = 1, sum(lst), 0
    # TODO: hardcoded path badness
    file_name = f"./tmp/{lst[0]}_{lst[1]}_{lst[2]}_{lst[3]}.xml"
    res = ""

    while lo <= hi:
        mid = (lo + hi) // 2

        with open(file_name, 'w') as file:
            printGraph(lst, mid, file)

        res = getRes(file_name, kiter_path)

        if "inf" in res:
            lo = mid + 1
        else:
            hi = mid - 1

    while "inf" in res:
        mid += 1
        with open(file_name, 'w') as file:
            printGraph(lst, mid, file)
        res = getRes(file_name, kiter_path)

    return [lst[0], lst[1], lst[2], lst[3], mid]
    # with open("results.txt", "a") as file:
    #     result = f"{lst[0]} {lst[1]} {lst[2]} {lst[3]} {mid}\n"
    #     file.write(result)


def gen_dict(limit, kiter_bin_path):
    result = generate_combinations(limit)
    # with open("data_no_res.txt", 'w') as file:
    #     for entry in result:
    #         file.write(f'{entry[0]},{entry[1]},{entry[2]},{entry[3]}\n')
    # for lst in result:
    #     getFlow(lst, kiter_bin_path)

    partial_get_flow = partial(getFlow, kiter_bin_path)
    # Create a Pool with the desired number of processes
    num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    # Apply the function to each item in parallel
    results = pool.map(partial_get_flow, result)

    # Close the pool to free resources
    pool.close()
    pool.join()

    with open("results.txt", "w") as file:
        for result in results:
            cs_res = f"{result[0]},{result[1]},{result[2]},{result[3]},{result[4]}\n"
            file.write(cs_res)
