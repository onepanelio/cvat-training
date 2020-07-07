import os
import sys
import shutil
import urllib.request
import tarfile
from datetime import datetime
import boto3

time = datetime.now()
stamp = time.strftime("%m%d%Y%H%M%S")


# parse parameters
# sample: epochs=100;num_classes=1
print("Arguments: ", sys.argv[1])
extra = sys.argv[1]
params = {}
for item in sys.argv[1].split(","):
	temp = item.split("=")
	if len(temp) == 2:
		params[temp[0].strip()] = temp[1].strip()

if 'num_clones' not in params:
	params['num_clones'] = 1
print("params: ", params)

if not os.path.exists("/mnt/data/models"):
	os.makedirs("/mnt/data/models")

#check if base model exists, if not then download
if params['ref-model-path'] == "":
	print("base model does not exist, downloading...")

	urllib.request.urlretrieve("https://github.com/onepanelio/templates/releases/download/v0.2.0/{}.tar".format(params['model']), "/mnt/data/models/model.tar")
	model_files = tarfile.open("/mnt/data/models/model.tar")
	model_files.extractall("/mnt/data/models")
	model_files.close()
	model_dir = "/mnt/data/models/"+params['model']
	files = os.listdir(model_dir)
	for f in files:
		shutil.move(model_dir+"/"+f,"/mnt/data/models")

else:
	s3_resource = boto3.resource('s3')
	bucket = s3_resource.Bucket(os.getenv('AWS_BUCKET_NAME')) 
	for object in bucket.objects.filter(Prefix = params['ref-model-path']):
		print("object key", object.key)
		bucket.download_file(object.key,'/mnt/data/models/'+os.path.basename(object.key))


os.chdir("/mnt/data/models")
print("print files")
print(os.system("ls"))
print("os listdir")
print(os.listdir())

os.system("pip install test-generator")
os.system("mkdir -p /mnt/src/protoc")
os.system("wget -P /mnt/src/protoc https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protoc-3.10.1-linux-x86_64.zip")
os.chdir("/mnt/src/protoc/")
os.system("unzip protoc-3.10.1-linux-x86_64.zip")
os.chdir("/mnt/src/tf/research/")
os.system("/mnt/src/protoc/bin/protoc object_detection/protos/*.proto --python_out=.")
from create_pipeline_v2 import create_pipeline

if "ssd-mobilenet-v2-coco" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 15000
	create_pipeline("/mnt/data/models/pipeline.config","/mnt/data/models/model.ckpt", params['dataset']+'/label_map.pbtxt', params['dataset']+'/*.tfrecord', params['dataset']+'/default.tfrecord', "/mnt/output/pipeline.config", params['epochs'],params['num_classes'], params['num_clones'],"ssd", params)


elif "ssd-mobilenet-v1-coco2" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 15000
	create_pipeline("/mnt/data/models/pipeline.config","/mnt/data/models/model.ckpt", params['dataset']+'/label_map.pbtxt', params['dataset']+'/*.tfrecord', params['dataset']+'/default.tfrecord', "/mnt/output/pipeline.config", params['epochs'],params['num_classes'], params['num_clones'],"ssd", params)


elif "frcnn-res101-coco" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 10000
	create_pipeline("/mnt/data/models/pipeline.config","/mnt/data/models/model.ckpt", params['dataset']+'/label_map.pbtxt', params['dataset']+'/*.tfrecord', params['dataset']+'/default_val.tfrecord', "/mnt/output/pipeline.config", params['epochs'],params['num_classes'], params['num_clones'],"frcnn", params)


elif "frcnn-res50-low" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 10000
	create_pipeline("/mnt/data/models/pipeline.config","/mnt/data/models/model.ckpt", params['dataset']+'/label_map.pbtxt', params['dataset']+'/*.tfrecord', params['dataset']+'/default.tfrecord', "/mnt/output/pipeline.config", params['epochs'],params['num_classes'], params['num_clones'],"frcnn", params)
elif "frcnn-res50-coco" in params['model'] or "faster-rcnn-res50" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 10000
	create_pipeline("/mnt/data/models/pipeline.config","/mnt/data/models/model.ckpt", params['dataset']+'/label_map.pbtxt', params['dataset']+'/*.tfrecord', params['dataset']+'/default.tfrecord', "/mnt/output/pipeline.config", params['epochs'],params['num_classes'], params['num_clones'],"frcnn", params)

elif "frcnn-res101-low" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 10
	create_pipeline("/mnt/data/models/pipeline.config","/mnt/data/models/model.ckpt", params['dataset']+'/label_map.pbtxt', params['dataset']+'/*.tfrecord', params['dataset']+'/default.tfrecord', "/mnt/output/pipeline.config", params['epochs'],params['num_classes'], params['num_clones'],"frcnn", params)


elif "frcnn-nas-coco" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 10
	create_pipeline("/mnt/data/models/pipeline.config","/mnt/data/models/model.ckpt", params['dataset']+'/label_map.pbtxt', params['dataset']+'/*.tfrecord', params['dataset']+'/default.tfrecord', "/mnt/output/pipeline.config", params['epochs'],params['num_classes'], params['num_clones'],"frcnn", params)

elif "ssdlite-mobilenet-coco" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 10
	create_pipeline("/mnt/data/models/pipeline.config","/mnt/data/models/model.ckpt", params['dataset']+'/label_map.pbtxt', params['dataset']+'/*.tfrecord', params['dataset']+'/default.tfrecord', "/mnt/output/pipeline.config", params['epochs'],params['num_classes'], params['num_clones'],"ssd", params)

os.chdir("/mnt/output")
os.mkdir("eval/")



os.system("python /mnt/src/tf/research/object_detection/legacy/train.py --train_dir=/mnt/output/ --pipeline_config_path=/mnt/output/pipeline.config --num_clones={}".format(params['num_clones']))
os.system("python /mnt/src/tf/research/object_detection/export_inference_graph.py --input-type=image_tensor --pipeline_config_path=/mnt/output/pipeline.config --trained_checkpoint_prefix=/mnt/output/model.ckpt-{} --output_directory=/mnt/output".format(params["epochs"]))


#generate lable map
os.system("python /mnt/src/train/convert_json_workflow.py {}/".format(params['dataset']))

print("*** Uploading Trained Model To Bucket Name: ***", os.getenv('AWS_BUCKET_NAME'))

#evaluate model in the end
# this is commented because the v1.13.0 of TF OD API is older uses unicode instead of str and updating it might break other parts.
# either fork that repo and update the file or update tf model and fix other parts that it might break
#os.system("python /mnt/src/tf/research/object_detection/legacy/eval.py --checkpoint_dir=/mnt/output/ --pipeline_config_path=/mnt/output/pipeline.config --eval_dir=/mnt/output/eval/")
