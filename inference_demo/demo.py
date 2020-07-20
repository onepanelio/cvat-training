

import os
import tensorflow.compat.v1 as tf
import numpy as np
import json
import ast
import cv2
import argparse
from PIL import Image
import math
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

    @staticmethod
    def process_boxes(boxes, scores, classes, labels_mapping, threshold, width, height):
        result = {}
        for i in range(len(classes[0])):
            if classes[0][i] in labels_mapping.keys():
                if scores[0][i] >= threshold:
                    xmin = int(boxes[0][i][1] * width)
                    ymin = int(boxes[0][i][0] * height)
                    xmax = int(boxes[0][i][3] * width)
                    ymax = int(boxes[0][i][2] * height)
                    label = labels_mapping[classes[0][i]]
                    if label not in result:
                        result[label] = []
                    result[label].append([xmin,ymin,xmax,ymax])
        return result

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
    
    def get_polygons(self, images, threshold):
        res = self.model.detect(images)
        result = {}
        for r in res:
            for index, c_id in enumerate(r['class_ids']):
                if c_id in self.labels_mapping.keys():
                    if r['scores'][index] >= threshold:
                        mask = r['masks'][:,:,index].astype(np.uint8)
                        contours = find_contours(mask, 0.5)
                        contour = contours[0]
                        contour = np.flip(contour, axis=1)
                        contour = approximate_polygon(contour, tolerance=2.5)
                        segmentation = contour.ravel().tolist()
                        label = self.labels_mapping[c_id]
                        if label not in result:
                            result[label] = []
                        result[label].append(segmentation)
        return result
    


    
    @staticmethod
    def process_polygons(polygons, boxes):
        def _check_inside_boxes(polygon, boxes):
            for point in polygon:
                for label, bxes in boxes.items():
                    for box in bxes:
                        if point[0] > box[0] and point[0] < box[2] and point[1] > box[1] and point[1] < box[3] and label not in ['dead','non_recoverable']:
                            # point is inside rectangle
                            return True
            return False
    
        result = {}
        for label_m, polys in polygons.items():
            for polygon in polys:
                p = [polygon[i:i+2] for i in range(0, len(polygon),2)]
                if _check_inside_boxes(p, boxes):
                    if label_m not in result:
                        result[label_m] = []
                    result[label_m].append(polygon)
                
        return result


def load_image_into_numpy(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def process_boxes(boxes, scores, classes, labels_mapping, threshold, width, height):
    result = {}
    for i in range(len(classes[0])):
        if classes[0][i] in labels_mapping.keys():
            if scores[0][i] >= threshold:
                xmin = int(boxes[0][i][1] * width)
                ymin = int(boxes[0][i][0] * height)
                xmax = int(boxes[0][i][3] * width)
                ymax = int(boxes[0][i][2] * height)
                label = labels_mapping[classes[0][i]]
                if label not in result:
                    result[label] = []
                result[label].append([xmin,ymin,xmax,ymax])
    return result


def draw_instances(frame, boxes, masks):
    colors = {'zero':(0,255,0), 'light':(0,0,255),'medium':(255,0,0),'high':(120,120,0),'non_recoverable':(0,120,120),'cut':(0,0,0)}
    for label, bxes in boxes.items():
        for box in bxes:
            cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), colors[label], 5)
    for label, polygons in masks.items():
        for polygon in polygons:
            p = [polygon[i:i+2] for i in range(0, len(polygon),2)]
            pts = np.array(p, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame, [pts], True, (0,255,255),5)
    return frame
    
def main(args):
    od_model = ObjectDetection("./frozen_inference_graph.pb")
    seg_model = Segmentation("./mask_rcnn_cvat_0160.h5")
    cap = cv2.VideoCapture(args.video)
    #would be better to take csv files as an input
    labels_mapping_od = {1:'zero',2:'light',3:'medium',4:'high',5:'non_recoverable'}
    frame_no = 0 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output_video.mp4', fourcc, math.ceil(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while True:
        ret, frame = cap.read()
        if ret:
            frame_no += 1
            print(frame_no)
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
            #normalize bounding boxes, also apply threshold
            od_result = ObjectDetection.process_boxes(boxes, scores, classes, labels_mapping_od, args.od_threshold, width, height)
            print(boxes)

            print("od", od_result)
            # run segmentation
            result = seg_model.get_polygons([image_np], args.mask_threshold)
            print("result", result)
            result = Segmentation.process_polygons(result, od_result)
            frame = draw_instances(frame, od_result, result)

            #write video
            out.write(frame)
            if frame_no == 15:
                cap.release()
                out.release()
                break
        else:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="path to video")
    parser.add_argument("--label_map",help="path to classes.csv")
    parser.add_argument("--model",help="path to trained model")
    parser.add_argument("--od_threshold",type=float, default=0.5, help="threshold for IoU")
    parser.add_argument("--mask_threshold",type=float, default=0.5, help="threshold for maskrcnn")
    parser.add_argument("--start_frame", type=int, default=0, help="when to start inference")

    parser.add_argument("--stop_frame", type=int, default=10, help="when to stop inference")
    parser.add_argument("--target", help="path to json file containing groundtruth data")
    parser.add_argument("--f",type=bool, default=False, help="run inference even though json file is present")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()
    #generate class mapping
    main(args)