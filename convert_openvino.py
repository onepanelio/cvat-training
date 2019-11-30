import os
import onepanel
from convert_json import converter

# TODO: change script path dynamically
os.system("python mo_tf.py --input_model=/onepanel/output/frozen_inference_graph.pb --tensorflow_use_custom_operations_config=extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config=/onepanel/output/pipeline.config")
#generate lable map
# converter(path_to_dataset)
os.mkdir("model_output")
os.system("mv frozen_inference_graph.bin model_output/")
os.system("mv frozen_inference_graph.xml model_output/")
#if "ssd" in model_name
os.system("mv /onepanel/code/interp_scripts/ssd_interp.py model_output/")
os.
os.chdir("model_output")


