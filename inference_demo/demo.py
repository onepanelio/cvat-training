

import os
import tensorflow.compat.v1 as tf
import numpy as np
import json
import ast
import cv2
import argparse
from PIL import Image
import sys
sys.path.append(os.environ.get('AUTO_SEGMENTATION_PATH')) 
from mrcnn.config import Config
import mrcnn.model as modellib
#from visualize import display_instances
import skimage.io
from skimage.measure import find_contours, approximate_polygon

# read video
# load all models
# for each frame
#   run object detection(s)
#   run semantic segmentation
#   perform post processing
#   write to video



   
class ObjectDetection:
    def __init__(self, model_path):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path , 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True
                self.sess = tf.Session(graph=self.detection_graph, config=config)

    def get_detections(self, image_np_expanded):
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
        return boxes, scores, classes, num_detections

class Segmentation:
    def __init__(self, model_path, num_c=2):
        
        class InferenceConfig(Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            NAME = "cvat"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = num_c

        config = InferenceConfig()
        #config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir="./output", config=config)
        # Load weights trained on MS-COCO
        self.model.load_weights(model_path, by_name=True)
        self.labels_mapping = {0:'BG', 1:'cut'}
    
    def get_polygons(self, images):
        res = model.detect(images)
        for r in res:
            for index, c_id in enumerate(r['class_ids']):
                if c_id in self.labels_mapping.keys():
                    if r['scores'][index] >= treshold:
                        mask = _convert_to_int(r['masks'][:,:,index])
                        segmentation = _convert_to_segmentation(mask)
                        label = self.labels_mapping[c_id]
                        if label not in result:
                            result[label] = []
                        result[label].append(
                            [image_num, segmentation])

def main(args):
    od_model = ObjectDetection("/data/frozen_inference_graph.pb")
    seg_model = Segmentation("/data/mask_rcnn_cvat_0160.h5")
    cap = cv2.VideoCapture(args.video)
    while True:
        ret, frame = cap.read()
        if ret:
            # get image ready for inference
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            width, height = image.size
            if width > 1920 or height > 1080:
                image = image.resize((width // 2, height // 2), Image.ANTIALIAS)
            image_np = load_image_into_numpy(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # run detection
            boxes, scores, classes, num_detections = od_model.get_detections(image_np_expanded)

            # run segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="path to video")
    parser.add_argument("--label_map",help="path to classes.csv")
    parser.add_argument("--model",help="path to trained model")
    parser.add_argument("--threshold",type=float, default=0.5, help="threshold for IoU")
    parser.add_argument("--eval", type=bool, default=True, help="should evaluate model or not")
    parser.add_argument("--start_frame", type=int, default=0, help="when to start inference")

    parser.add_argument("--stop_frame", type=int, default=10, help="when to stop inference")
    parser.add_argument("--target", help="path to json file containing groundtruth data")
    parser.add_argument("--f",type=bool, default=False, help="run inference even though json file is present")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()
    #generate class mapping
    main(args)