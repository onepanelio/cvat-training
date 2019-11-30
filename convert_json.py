import json
import tensorflow as tf

from google.protobuf import text_format

with open("sample.pbtxt",'r') as f:
	txt = f.readlines()

data = {"label_map":{}}
for line in txt:

	if "id" in line:
		i = str(line.split(":")[1].strip())
		data["label_map"][i] = ''
	if "display_name" in line:
		n = line.split(":")[1].strip()
		print(n)
		data["label_map"][i] = n.replace('\\"', "\"")


with open("sample_out.json","w") as out:
	json.dump(data, out)
