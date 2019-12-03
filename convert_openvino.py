import os
from random import randint
import sys
# import onepanel
# from convert_json import converter


# TODO: mv to model_optimizer in bash
# TODO: change script path dynamically

# Check if dataset already exists
os.chdir("/onepanel/code/dldt/model-optimizer/")
if "ssd" in sys.argv[2]:
	os.system("python mo_tf.py --input_model=/onepanel/output/frozen_inference_graph.pb --tensorflow_use_custom_operations_config=extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config=/onepanel/output/pipeline.config")
elif "faster" in sys.argv[2]:
	os.system("python mo_tf.py --input_model=/onepanel/output/frozen_inference_graph.pb --tensorflow_use_custom_operations_config=extensions/front/tf/faster_rcnn_support_api_v1.10.json --tensorflow_object_detection_api_pipeline_config=/onepanel/output/pipeline.config")

#generate lable map
os.system("python /onepanel/code/convert_json_2.py {}/".format(sys.argv[1]))
dataset_name = "modeloutput{}".format(randint(1000000000, 2000000000))
os.system("onepanel datasets create {}".format(dataset_name))
os.system("mv /onepanel/code/dldt-2018_R5/model-optimizer/frozen_inference_graph.bin /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))
os.system("mv /onepanel/code/dldt-2018_R5/model-optimizer/frozen_inference_graph.xml /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))
if "ssd" in sys.argv[2]:
	os.system("mv /onepanel/code/interp_scripts/ssd_interp.py /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))
elif "faster" in sys.argv[2]:
	os.system("mv /onepanel/code/interp_scripts/faster_rcnn.py /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))

os.system("mv /onepanel/code/dldt-2018_R5/model-optimizer/label_map.json /onepanel/code/dldt-2018_R5/model-optimizer/{}/".format(dataset_name))
os.chdir("/onepanel/code/dldt-2018_R5/model-optimizer/{}".format(dataset_name))
os.system("mv /onepanel/code/dldt-2018_R5/model-optimizer/modeloutput{}/ssd_interp.py /onepanel/code/dldt-2018_R5/model-optimizer/modeloutput{}/interp.py".format(dataset_name, dataset_name))
os.system('onepanel datasets push -m "update"')
#


