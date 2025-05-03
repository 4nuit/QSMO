#!/usr/bin/python

import sys

if len(sys.argv) != 2:
	print("Usage: python script.py <filename>")
	sys.exit(1)

try:
	with open(sys.argv[1], 'rb') as f:
		r = f.readlines()
		proba_sum = sum([int(e.strip()) for e in r])
		print(f"Proba sum: {proba_sum}")

except FileNotFoundError:
	print(f"Error: File '{sys.argv[1]}' not found.")
	sys.exit(1)
