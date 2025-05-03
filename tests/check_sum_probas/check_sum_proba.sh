#!/bin/bash

cat result.txt |cut -d ":"  -f 2 |cut -d " " -f 2 |cut -d "%" -f 1 | bc > file.txt
./test.py file.txt
