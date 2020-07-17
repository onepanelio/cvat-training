

import os
import tensorflow as tf
import numpy as np
import json
import ast
import cv2
import argparse
from PIL import Image
from evals import compute_confusion_matrix
from evals import print_cm

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