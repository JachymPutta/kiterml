#!/bin/bash
LIMIT=30

for i in $(seq 1 $LIMIT);
do
    ./binary_search.lua $i $i &
done

wait