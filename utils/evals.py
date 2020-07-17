import numpy as np
import pandas as pd

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
 
    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))
  
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