import json
import tensorflow as tf
import os
from google.protobuf import text_format
import sys

# def converter(path):
# 	print("reading pbtxt file...", os.path.join(path, "label_map.pbtxt"))
# 	with open(os.path.join(path, "label_map.pbtxt"),'r') as f:
# 		txt = f.readlines()
# 	print("generating label_map.json file...")
# 	f_out = open(os.path.join("/onepanel/output/", "label_map.json"),"w")
# 	f_out.write('{ "label_map": { \n')
# 	for line in txt:

# 		if "id" in line:
# 			i = str(line.split(":")[1].strip())
# 			# data["label_map"][i] = ''
# 			f_out.write('"{}":'.format(i))
# 		if "name"  in line:
# 			n = line.split(":")[1].strip()
# 			# print(n)
# 			f_out.write("{}, \n".format(n))
# 			# data["label_map"][i] = n.replace('\\"', "\"")
# 	print("writing json file...")
# 	f_out.write("}}")
# 	data = open("label_map.json", "r").read()
# 	print(data)

# def converter(path):
# 	print("reading pbtxt file...", os.path.join(path, "label_map.pbtxt"))
# 	with open(os.path.join(path, "label_map.pbtxt"),'r') as f:
# 		txt = f.readlines()
# 	print("generating label_map.json file...")
#   f_out = open(os.path.join("label_map.json"),"w")
#   f_out.write('{ "label_map": { \n')
# 	data = {}
# 	for line in txt:

# 		if "id" in line:
# 			i = str(line.split(":")[1].strip())
# 			# data["label_map"][i] = ''
# #           f_out.write('"{}":'.format(i))
# 			data[i] = None
# 		if "name"  in line:
# 			n = line.split(":")[1].strip().strip("'")
# 			# print(n)
            
# 			data[i] = n
# #           f_out.write('"{}",\n'.format(n))
# 			# data["label_map"][i] = n.replace('\\"', "\"")
# 	# print(data)
# 	d = {"label_map":data}
# 	with open(os.path.join("/onepanel/output/", "label_map.json"), 'w') as outfile:
# 		json.dump(d, outfile)
		

import csv

def converter_withcsv(path):
	print("reading pbtxt file...", os.path.join(path, "label_map.pbtxt"))
	with open(os.path.join(path, "label_map.pbtxt"),'r') as f:
		txt = f.readlines()
	print("generating label_map.json file...")
	csv_out = open(os.path.join("/onepanel/output/", "classes.csv"), "w")
	csv_writer = csv.writer(csv_out)
	csv_writer.writerow(['labels'])
#   f_out = open(os.path.join("label_map.json"),"w")
#   f_out.write('{ "label_map": { \n')
	data = {}
	for line in txt:

		if "id" in line:
			i = str(line.split(":")[1].strip())
			# data["label_map"][i] = ''
#           f_out.write('"{}":'.format(i))
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
