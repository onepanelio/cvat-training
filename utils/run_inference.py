

import os

import tensorflow as tf
import pandas as pd
import numpy as np
import json
import ast
import cv2
import argparse
from PIL import Image


def load_image_into_numpy(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def generate_labels(csv_file):
    mapping = {}
    with open(csv_file, "r") as f:
        f.readline()  # First line is header
        line = f.readline().rstrip()
        cnt = 1
        while line:
            mapping[cnt] = line
            line = f.readline().rstrip()
            cnt += 1
    print("Classes: ", mapping)
    return mapping


def run_tensorflow_annotation(video, model_path, threshold, labels_mapping, start_frame, stop_frame):
    def _normalize_box(box, w, h):
        xmin = int(box[1] * w)
        ymin = int(box[0] * h)
        xmax = int(box[3] * w)
        ymax = int(box[2] * h)
        return xmin, ymin, xmax, ymax

    cap = cv2.VideoCapture(video)
    result = {}
    count_frames = 0
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path , 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        try:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(graph=detection_graph, config=config)
            while True:
                ret, frame = cap.read()
                if ret:
                    print("Processing frame: ", count_frames)
                    if count_frames < start_frame:
                        count_frames += 1
                        continue
                    if count_frames > stop_frame:
                        break
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(img)
                    width, height = image.size
                    if width > 1920 or height > 1080:
                        image = image.resize((width // 2, height // 2), Image.ANTIALIAS)
                    image_np = load_image_into_numpy(image)
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
                    temp = {'boxes':[], 'scores':[], 'class_ids':[], }
                    for i in range(len(classes[0])):
                        if classes[0][i] in labels_mapping.keys():
                            if scores[0][i] >= threshold:
                                xmin, ymin, xmax, ymax = _normalize_box(boxes[0][i], width, height)
                                label = labels_mapping[classes[0][i]]
                                temp['boxes'].append([xmin, ymin, xmax,ymax])
                                temp['scores'].append(scores[0][i])
                                temp['class_ids'].append(int(classes[0][i]))
                                
                                result[count_frames] = temp
                    count_frames += 1
                    
                else:
                    print("Finished processing...")
        finally:
            sess.close()
            del sess
    # print("Result: ", result)
    return result

def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())
    
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def compute_confusion_matrix(result, target, categories, start_frame, stop_frame, IOU_THRESHOLD):
    # record_iterator = tf.python_io.tf_record_iterator(path=detections_record)
    # data_parser = tf_example_parser.TfExampleDetectionAndGTParser()

    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))
    # print(target[687])
    # print(result[687])
    image_index = 0
    for example_no in list(target.keys()):
        if example_no < start_frame or example_no > stop_frame:
            break
        image_index += 1
        if example_no in target:
            groundtruth_boxes = np.asarray(target[example_no]['boxes'])
            groundtruth_classes = np.asarray(target[example_no]['class_ids'])
        else:
            groundtruth_boxes = np.asarray([])
            groundtruth_classes = np.asarray([])
        if example_no in result:
            detection_classes = np.asarray(result[example_no]['class_ids'])
            detection_boxes = np.asarray(result[example_no]['boxes'])
        else:
            detection_classes = np.asarray([])
            detection_boxes = np.asarray([])
        
        matches = []

        if image_index % 100 == 0:
            print("Processed %d images" %(image_index))
        
        for i in range(len(groundtruth_boxes)):
            for j in range(len(detection_boxes)):
                iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])
                if iou > IOU_THRESHOLD:
                    matches.append([i, j, iou])
                
        matches = np.array(matches)
        if matches.shape[0] > 0:
            # Sort list of matches by descending IOU so we can remove duplicate detections
            # while keeping the highest IOU entry.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
            
            # Remove duplicate detections from the list.
            matches = matches[np.unique(matches[:,1], return_index=True)[1]]
            
            # Sort the list again by descending IOU. Removing duplicates doesn't preserve
            # our previous sort.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
            
            # Remove duplicate ground truths from the list.
            matches = matches[np.unique(matches[:,0], return_index=True)[1]]
            
        for i in range(len(groundtruth_boxes)):
            if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
                confusion_matrix[groundtruth_classes[i] - 1][detection_classes[int(matches[matches[:,0] == i, 1][0])] - 1] += 1 
            else:
                confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
                
        for i in range(len(detection_boxes)):
            if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
                confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1
    

    print("Processed %d images" % (image_index))

    return confusion_matrix

def print_cm(confusion_matrix, categories, IOU_THRESHOLD):
    print("Confusion Matrix:")
    print(confusion_matrix, "\n")
    results = []

    for i in range(len(categories)):
        id = list(categories.keys())[i]-1
        name = list(categories.values())[i]
        
        total_target = np.sum(confusion_matrix[id,:])
        total_predicted = np.sum(confusion_matrix[:,id])
        if int(total_predicted) == 0:
            precision = 0
        else:
            precision = float(confusion_matrix[id, id] / total_predicted)
        if int(total_target) == 0:
            recall = 0
        else:
            recall = float(confusion_matrix[id, id] / total_target)
        
        results.append({'category' : name, 'precision_@{}IOU'.format(IOU_THRESHOLD) : precision, 'recall_@{}IOU'.format(IOU_THRESHOLD) : recall})
    print("Output: ")
    print(pd.DataFrame(results))

def parse_gt(target_file):
    with open(target_file, "r") as file:
        groundtruth = json.load(file)
    lm = {}
    for i in groundtruth['categories']:
        lm[i['id']] = i['name']
    data = {}
    for i in groundtruth['annotations']:
        if i['image_id'] not in data:
            data[i['image_id']] = {'boxes':[[i['bbox'][0],i['bbox'][1],i['bbox'][0]+i['bbox'][2] ,i['bbox'][1]+i['bbox'][3]]], 'class_ids':[i['category_id']]}
        else:
            data[i['image_id']]['boxes'].append([i['bbox'][0],i['bbox'][1] ,i['bbox'][0]+i['bbox'][2] ,i['bbox'][1]+i['bbox'][3]])   
            data[i['image_id']]['class_ids'].append(i['category_id'])
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", help="path to video")
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
    labels_mapping = generate_labels(args.label_map)
    output_path = os.path.basename(args.input_video)[:-4]+"_"+str(args.start_frame)+"_"+str(args.stop_frame)+"_result.json"
    if not os.path.exists(output_path) or args.f:
        print("Running inference...")
        result = run_tensorflow_annotation(args.input_video, args.model, args.threshold, labels_mapping, args.start_frame, args.stop_frame)
        with open(output_path, "w") as file:
            file.write(str(result))
    else:
        print("Loading annotation from file...")
        #read result from file
        with open(output_path, "r") as file:
            result = ast.literal_eval(file.read())
    # print("Result: ", result)
    if args.eval:
        groundtruth = parse_gt(args.target)
        # print("groundtruth 0", groundtruth[0], groundtruth[1], groundtruth[9])
        cm = compute_confusion_matrix(result, groundtruth, labels_mapping, args.start_frame, args.stop_frame, args.iou_threshold)
        print_cm(cm, labels_mapping, args.iou_threshold)