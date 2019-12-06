import os
import sys
import uuid

# parse parameters
# sample: epochs=100;num_classes=1
params = {}
for item in sys.argv[1].split(","):
	temp = item.split("=")
	params[temp[0].strip()] = temp[1].strip()

if 'num_clones' not in params:
	params['num_clones'] = 1

#TODO: add param for decays

# epochs
# num_classe
# dataset
# model
#print(params)
# ssd-mobilenet-v2-coco  (Tested)
#faster-rcnn-resnet50-lowp (tested)
#faster-rcnn-resnet101-coc (tested)
#          resnet101-low (tested)
#          nas-coco-2018  (tested)
#         nas-lowpropos   (tested)
#         mask-rcnn-inception-resne(t) (failed)
#         mask-rcnn-inception-v2-co (tested)
#         mask-rcnn-resnet101-atrou (tested)
#
os.system("pip install test-generator")
os.system("wget https://github.com/opencv/dldt/archive/2018_R5.zip")
os.system("unzip 2018_R5.zip")
#os.system("git clone https://github.com/tensorflow/models.git /onepanel/extra_repos/tensorflow_models")
os.system("mkdir -p /onepanel/bin/protoc")
os.system("wget -P /onepanel/bin/protoc https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protoc-3.10.1-linux-x86_64.zip")
os.chdir("/onepanel/bin/protoc/")
os.system("unzip protoc-3.10.1-linux-x86_64.zip")
os.chdir("/onepanel/extra_repos/tensorflow_models/research/")
#os.system("export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim")
os.system("/onepanel/bin/protoc/bin/protoc object_detection/protos/*.proto --python_out=.")
os.chdir(params['dataset'])
os.system('latest=$(find . -name "*.tfrecord*.zip" -print0 | xargs -r -0 ls -1 -t | head -n1) && unzip -o "$latest"')
if "ssd" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 30000
	os.system("python /onepanel/code/create_pipeline.py -in_pipeline /onepanel/input/datasets/aleksandr-cluster0-01/ssd-mobilenet-v2-coco-201/2/pipeline.config -num_classes {} -epochs {} -model /onepanel/input/datasets/aleksandr-cluster0-01/ssd-mobilenet-v2-coco-201/2/model.ckpt -label {}/label_map.pbtxt -train_data {}/training.tfrecord -eval_data {}/training.tfrecord -out_pipeline /onepanel/output/pipeline.config".format(params["num_classes"], params["epochs"], params["dataset"], params["dataset"], params["dataset"]))
elif "faster-rcnn-resnet101" in params['model']:
	if 'epochs' not in params:
		params['epochs'] = 100000
	os.system("python /onepanel/code/create_pipeline_v2.py -in_pipeline /onepanel/input/datasets/san999/faster-rcnn-resnet101-coco/1/pipeline.config -num_classes {} -epochs {} -model /onepanel/input/datasets/san999/faster-rcnn-resnet101-coco/1/model.ckpt -label {}/label_map.pbtxt -train_data {}/training.tfrecord -eval_data {}/training.tfrecord -out_pipeline /onepanel/output/pipeline.config".format(params["num_classes"], params["epochs"], params["dataset"], params["dataset"], params["dataset"]))
elif "faster-rcnn-resnet50" in params['model']:
	if 'epochs' not in params:
		paramsp['epochs'] = 100000
	os.system("python /onepanel/code/create_pipeline_v2.py -in_pipeline /onepanel/input/datasets/joinalop/faster-rcnn-resnet50-lowp/1/pipeline.config -num_classes {} -epochs {} -model /onepanel/input/datasets/joinalop/faster-rcnn-resnet50-lowp/1/model.ckpt -label {}/label_map.pbtxt -train_data {}/training.tfrecord -eval_data {}/training.tfrecord -out_pipeline /onepanel/output/pipeline.config".format(params["num_classes"], params["epochs"], params["dataset"], params["dataset"], params["dataset"]))


os.system("python /onepanel/extra_repos/tensorflow_models/research/object_detection/legacy/train.py --train_dir=/onepanel/output/ --pipeline_config_path=/onepanel/output/pipeline.config --num_clones={}".format(params['num_clones']))
os.system("python /onepanel/extra_repos/tensorflow_models/research/object_detection/export_inference_graph.py --input-type=image_tensor --pipeline_config_path=/onepanel/output/pipeline.config --trained_checkpoint_prefix=/onepanel/output/model.ckpt-{} --output_directory=/onepanel/output".format(params["epochs"]))


### python convert_openvino.py /onepanel/input/datasets/aleksandr-cluster0-01/test3-4-car4-f3f6ab924c074a4c89e55957d7dc51ab196e3307e4924f98a3ec7d9f1363a7cd/4 ssd-mobilenet-v2-coco-201

# TODO: mv to model_optimizer in bash
# TODO: change script path dynamically

# Check if dataset already exists
os.chdir("/onepanel/code/dldt-2018_R5/model-optimizer/")
if "ssd" in params['model']:
	os.system("python mo_tf.py --input_model=/onepanel/output/frozen_inference_graph.pb --tensorflow_use_custom_operations_config=/onepanel/code/ssd_support_api_v1.14.json --tensorflow_object_detection_api_pipeline_config=/onepanel/output/pipeline.config")
elif "faster-rcnn-resnet101" in params['model']:
	os.system("python mo_tf.py --input_model=/onepanel/output/frozen_inference_graph.pb --tensorflow_use_custom_operations_config=extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api_pipeline_config=/onepanel/output/pipeline.config")
elif "faster-rcnn-resnet50-lowp" in params['model']:



#generate lable map
os.system("python /onepanel/code/convert_json_2.py {}/".format(params['dataset']))
dataset_name = "{}-model-output-{}".format(params['model'], uuid.uuid4().int)
os.system("onepanel datasets create {}".format(dataset_name))
os.system("mv /onepanel/code/dldt-2018_R5/model-optimizer/frozen_inference_graph.bin /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))
os.system("mv /onepanel/code/dldt-2018_R5/model-optimizer/frozen_inference_graph.xml /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))
if "ssd" in params['model']:
	os.system("mv /onepanel/code/interp_scripts/ssd_interp.py /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))
	os.system("mv /onepanel/code/dldt-2018_R5/model-optimizer/{}/ssd_interp.py /onepanel/code/dldt-2018_R5/model-optimizer/{}/interp.py".format(dataset_name, dataset_name))

elif "faster" in params['model']:
	os.system("mv /onepanel/code/interp_scripts/faster_rcnn.py /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))

os.system("mv /onepanel/output/label_map.json /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))
os.chdir("/onepanel/code/dldt-2018_R5/model-optimizer/{}".format(dataset_name))
os.system('onepanel datasets push -m "update"')
#


