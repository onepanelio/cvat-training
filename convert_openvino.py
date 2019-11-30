import os
from random import randint

# import onepanel
# from convert_json import converter


# TODO: mv to model_optimizer in bash
# TODO: change script path dynamically

# Check if dataset already exists
os.chdir("/onepanel/code/dldt-2018_R5/model-optimizer/")
os.system("python mo_tf.py --input_model=/onepanel/output/frozen_inference_graph.pb --tensorflow_use_custom_operations_config=extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config=/onepanel/output/pipeline.config")
#generate lable map
# os.system("python /onepanel/code/convert_json.py path_to_Dataset")
dataset_name = "modeloutput{}".format(randint(1000000000, 2000000000))
os.system("onepanel datasets create {}".format(dataset_name))
os.system("mv frozen_inference_graph.bin {}/".format(dataset_name))
os.system("mv frozen_inference_graph.xml {}/".format(dataset_name)
#if "ssd" in model_name
os.system("mv /onepanel/code/interp_scripts/ssd_interp.py {}/".format(dataset_name))
os.system("mv label_map.json {}/".format(dataset_name))
os.chdir("/onepanel/code/dldt-2018_R5/model-optimizer/{}".format(dataset_name))
os.system("mv ssd_interp.py interp.py")
os.system('onepanel datasets push -m "update"')
#


