#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import os

import constants

from generation.gen_data import gen_dict
from generation.gen_permutation_throughput import gen_permutations
from generation.gen_min_max_throughut import gen_min_max

parser = argparse.ArgumentParser(description="KiterML: data set generation tools")

parser.add_argument('--type', help='Type of data to generate',
                         choices=['dict', 'min_max', 'permutations'], required=True)
parser.add_argument('--file', help='Input data file - permutations and min-max only')
parser.add_argument('--kiter', help='Path to the kiter binary - dict only')
parser.add_argument('--limit', help='Limit tokens - dict only')
parser.add_argument('--out', help='Output file')

args = parser.parse_args()

if args.type == "dict":
    gen_dict(int(args.limit), args.kiter)
elif args.type == "min_max":
    gen_min_max(args.file, args.out)
elif args.type == "permutations":
    gen_permutations(args.file, args.out)

