import json
import tensorflow as tf
import os
from google.protobuf import text_format
import sys

def converter(path):
	with open(os.path.join(path, "label_map.pbtxt"),'r') as f:
		txt = f.readlines()

	f_out = open(os.path.join(path, "label_map.json"),"w")
	f_out.write('{ "label_map": { \n')
	for line in txt:

		if "id" in line:
			i = str(line.split(":")[1].strip())
			# data["label_map"][i] = ''
			f_out.write('"{}":'.format(i))
		if "display_name" in line:
			n = line.split(":")[1].strip()
			# print(n)
			f_out.write("{}, \n".format(n))
			# data["label_map"][i] = n.replace('\\"', "\"")

	f_out.write("}}")

if __name__ == "__main__":
	converter(sys.argv[1])