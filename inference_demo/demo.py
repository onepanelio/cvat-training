import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import json
import ast
import cv2
import argparse
from PIL import Image
from gpslogger import GPSLogger
import math
import sys
sys.path.append(os.environ.get('AUTO_SEGMENTATION_PATH')) 
from mrcnn.config import Config
import mrcnn.model as modellib
import skimage.io
from skimage.measure import find_contours, approximate_polygon
   
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
        """
           Check if any point of the polygon falls into any of coconot palms except for dead/non_recoverable.
        """
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

def draw_instances(frame, boxes, masks):
    colors = {'zero':(0,255,0), 'light':(0,0,255),'medium':(255,0,0),'high':(120,120,0),'non_recoverable':(0,120,120),'cut':(0,0,0)}
    #draw boxes
    for label, bxes in boxes.items():
        for box in bxes:
            cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), colors[label], 5)
    #draw polygons
    for label, polygons in masks.items():
        for polygon in polygons:
            p = [polygon[i:i+2] for i in range(0, len(polygon),2)]
            pts = np.array(p, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame, [pts], True, (0,255,255),5)
    return frame
    
def main(args):
    if args.type == "both":
        od_model = ObjectDetection(args.od_model)
        seg_model = Segmentation(args.mask_model)
    elif args.type == "classes":
        od_model = ObjectDetection(args.od_model)
    elif args.type == "v_shape":
        seg_model = Segmentation(args.mask_model)
       
    cap = cv2.VideoCapture(args.video)
    #would be better to take csv files as an input
    #labels_mapping_od = {1:'dead', 2:'damaged',3:'healthy'}
    labels_mapping_od = {1:'zero',2:'light',3:'medium',4:'high',5:'non_recoverable'}
    frame_no = 0 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, math.ceil(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    #prepare GPS logger
    if args.gps_csv != None:
        gpsl = GPSLogger(args.video, args.gps_csv)

    final_result = {'boxes':{}, 'polygon':{}}
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame_no += 1
            print("Processing frame: ", frame_no)
            # get image ready for inference
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            width, height = image.size
            if width > 1920 or height > 1080:
                image = image.resize((width // 2, height // 2), Image.ANTIALIAS)
            image_np = load_image_into_numpy(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            od_result = {}
            result = {}
            if args.type == "both" or args.type == "classes":
                # run detection
                boxes, scores, classes, num_detections = od_model.get_detections(image_np_expanded)
                #normalize bounding boxes, also apply threshold
                od_result = ObjectDetection.process_boxes(boxes, scores, classes, labels_mapping_od, args.od_threshold, width, height)
                final_result['boxes'][frame_no] = od_result
            if args.type == "both" or args.type == "v_shape":
                # run segmentation
                result = seg_model.get_polygons([image_np], args.mask_threshold)
                if args.type == "both" or args.type == "classes":
                    # filter out false positives if boxes are available
                    result = Segmentation.process_polygons(result, od_result)
                final_result['polygon'][frame_no] = result
            

            frame = draw_instances(frame, od_result, result)

            #write video
            out.write(frame)

            # update features for geojson
            # gpsl.update_features(od_result, result, args.survey_type)

            if frame_no == 25:
                print(final_result)
                break
           
        else:
            try:
                if args.type == "both" or args.type == "classes":
                    final_result['boxes'][frame_no] = od_result
                if args.type == "both" or args.type == "v_shape":
                    final_result['polygon'][frame_no] = result
                with open(args.video.replace(".mp4","_model_output.json", "w")) as fl:
                    json.dump(final_result, fl)
                cap.release()
                out.release()
                break
            except:  #handle case when video is corrupted or does not exists
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",default="both",help="what type of models to use [both,classes,v_shape]")
    parser.add_argument("--video", help="path to video")
    parser.add_argument("--gps_csv", help="path to csv containing gps data")
    parser.add_argument("--od_model",help="path to trained detection model")
    parser.add_argument("--mask_model",help="path to trained maskrcnn model")
    parser.add_argument("--od_threshold",type=float, default=0.5, help="threshold for IoU")
    parser.add_argument("--mask_threshold",type=float, default=0.5, help="threshold for maskrcnn")
    parser.add_argument("--output_video", default="output_video.mp4", help="where to store output video")
    parser.add_argument("--survey_type", default="v_shape",help="what to write in geojson [v_shape,classes")
    args = parser.parse_args()
    if args.type not in ['both','classes','v_shape']:
        raise ValueError('Invalid type: {}. Valid options are "both","classes","v_shape".'.format(args.type))

    if not os.path.exists(args.video):
        raise FileExistsError("Video does not exist!")
    main(args)