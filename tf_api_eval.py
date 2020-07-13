import os
import sys
import shutil
import urllib.request
import tarfile
from datetime import datetime
import tensorflow as tf
import boto3


# parse parameters
# sample: epochs=100;num_classes=1
print("Arguments: ", sys.argv[1])
extra = sys.argv[1]
params = {}
for item in sys.argv[1].split(","):
	temp = item.split("=")
	if len(temp) == 2:
		params[temp[0].strip()] = temp[1].strip()

print("Eval Params: ", params)

os.system("pip install test-generator")
os.system("mkdir -p /mnt/src/protoc")
os.system("wget -P /mnt/src/protoc https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protoc-3.10.1-linux-x86_64.zip")
os.chdir("/mnt/src/protoc/")
os.system("unzip protoc-3.10.1-linux-x86_64.zip")
os.chdir("/mnt/src/tf/research/")
os.system("/mnt/src/protoc/bin/protoc object_detection/protos/*.proto --python_out=.")


if params['metrics_type'] == "tf-od-api":

	def count_ex(path):
		count = 0
		#there might be a better way to find length/num of examples
		for example in tf.python_io.tf_record_iterator(path):
			count += 1
		return count

	from create_pipeline_v2 import create_pipeline_eval
	count_examples = count_ex("/mnt/data/datasets/"+params['record_name'])
	if int(params['num_visualizations']) > count_examples:
    	#num visualizations should not be greater than num examples
		params['num_visualizations'] = count_examples
	create_pipeline_eval(count_examples, "/mnt/data/datasets/"+params['record_name'], params['tf_metrics_type'],int(params['num_visualizations']),"/mnt/data/models/pipeline.config")


	os.system("python /mnt/src/tf/research/object_detection/legacy/eval.py --checkpoint_dir=/mnt/data/models/ --pipeline_config_path=/mnt/data/models/pipeline_updated.config --eval_dir=/mnt/output/")

elif params['metrics_type'] == "confusion-matrix":
	# calculate confusion matrix
	os.system("python /mnt/src/tf/research/object_detection/inference/infer_detections.py --input_tfrecord_paths={} --output_tfrecord_path=/mnt/data/datasets/output.tfrecord --inference_graph=/mnt/data/models/frozen_inference_graph.pb".format("/mnt/data/datasets/"+params['record_name']))
	os.system("python /mnt/src/train/confusion_matrix.py --detections_record=/mnt/data/datasets/output.tfrecord --label_map=/mnt/data/datasets/label_map.pbtxt --output_path=/mnt/output/confusion_matrix.csv --iou_threshold={} --confidence_threshold={}".format(float(params['iou_threshold']), float(params['conf_threshold'])))
