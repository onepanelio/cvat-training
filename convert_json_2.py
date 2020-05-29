import json
import tensorflow as tf
import os
from google.protobuf import text_format
import sys


import csv

def converter_withcsv(path):
	print("reading pbtxt file...", os.path.join(path, "label_map.pbtxt"))
	with open(os.path.join(path, "label_map.pbtxt"),'r') as f:
		txt = f.readlines()
	print("generating label_map.json file...")
	csv_out = open(os.path.join("/onepanel/output/", "classes.csv"), "w")
	csv_writer = csv.writer(csv_out)
	csv_writer.writerow(['labels'])
	data = {}
	for line in txt:

		if "id" in line:
			i = str(line.split(":")[1].strip())
			# data["label_map"][i] = ''
			data[i] = None
		if "name"  in line:
			n = line.split(":")[1].strip().strip("'")
			# print(n)
			csv_writer.writerow([n])
			data[i] = n
	# print(data)
	d = {"label_map":data}
	with open(os.path.join("/onepanel/output/", "label_map.json"), 'w') as outfile:
		json.dump(d, outfile)



if __name__ == "__main__":
	converter_withcsv(sys.argv[1])
