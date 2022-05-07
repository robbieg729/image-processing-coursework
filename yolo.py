##########################################################################

# Example : performs YOLO (v3) object detection from a video file
# You should adapt this code for your purposes

# path to video specified on the command line - e.g. 
# python yolo.py --video=path_to_video

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2021 Amir Atapour Abarghouei

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Implements the You Only Look Once (YOLO) object detection architecture in:
# Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement.
# arXiv:1804.02767. - https://pjreddie.com/media/files/papers/YOLOv3.pdf

# This code: significant portions based in part on the tutorial and
# example available at:
# https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
# https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.py
# under LICENSE:
# https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/LICENSE

# To use, you need the weights, config file and class name
# All three are provided along with the code.
# You could also download the files, though using the provided
# files is recommended.

# https://pjreddie.com/media/files/yolov3.weights
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
# https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true

##########################################################################

import cv2
import argparse
import numpy as np

##########################################################################

keep_processing = True

# parse command line arguments for video file and YOLO files
parser = argparse.ArgumentParser(
    description='Perform YOLO object detection on video')
parser.add_argument(
    "-use",
    "--target",
    type=str,
    choices=['cpu', 'gpu', 'opencl'],
    help="select computational backend",
    default='cpu')
parser.add_argument(
    '--video_file',
    type=str,
    help='specify path to video file',
    required=True)
parser.add_argument(
    "-cl",
    "--class_file",
    type=str,
    help="list of classes",
    default='coco.names')
parser.add_argument(
    "-cf",
    "--config_file",
    type=str,
    help="network config",
    default='yolov3.cfg')
parser.add_argument(
    "-w",
    "--weights_file",
    type=str,
    help="network weights",
    default='yolov3.weights')

args = parser.parse_args()

#####################################################################

# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, confidence, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2f' % (class_name, confidence)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(
        image,
        (left,
         top -
         round(
             1.5 *
             labelSize[1])),
        (left +
         round(
             1.5 *
             labelSize[0]),
            top +
            baseLine),
        (255,
         255,
         255),
        cv2.FILLED)
    cv2.putText(image, label, (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

#####################################################################
# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression
# For the purposes of this code, threshold will be set at a minimum

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes
    # with lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        threshold_confidence,
        threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)

##########################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def writeScores(image, avg_conf, obj_count, total_count):
    height, width, channel = image.shape
    bottomLeftCorner = (10,height-15)
    bottomRightCorner = (width-210, height-15)
    bottomMiddleHigh = (width//2 - 270, height-65)
    bottomMiddleLow = (width//2 - 230, height-15)
    conf_text = '%s: %.2f' % ("Mean_Conf", avg_conf)
    count_text = '%s: %d' % ("Obj_Count", obj_count)
    score_eq = '%s' % ("Score = Mean(Mean_Conf, Obj_Count/Total_Count)")
    score_text = f"Score = ({avg_conf:.2f} + {obj_count}/{total_count})/2 = {(avg_conf + obj_count/total_count)/2:.3f}"
    cv2.putText(image, conf_text, bottomLeftCorner,
    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (123,49,126), 2)
    cv2.putText(image, count_text, bottomRightCorner,
    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (123,49,126), 2)
    cv2.putText(image, score_eq, bottomMiddleHigh,
    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (123,49,126), 2)
    cv2.putText(image, score_text, bottomMiddleLow,
    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (123,49,126), 2)
##########################################################################

# define video capture object

cap = cv2.VideoCapture()

output_video_file_name = 'output_video.avi'

frames = []

# total count for VALIDATION:
total_count = 391

# total count for TEST:
#total_count = 1475

##########################################################################

# init YOLO CNN object detection model

confThreshold = 0.01  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Load names of classes from file

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network
# using them

net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights_file)
output_layer_names = getOutputsNames(net)

# set up compute target as one of [GPU, OpenCL, CPU]

if (args.target == 'gpu'):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
elif (args.target == 'opencl'):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

##########################################################################
conf_count = 1
conf_sum = 0
# use command line arguments to read video_name
print("Reading video file")

if (((args.video_file) and (cap.open(str(args.video_file))))):

    while (keep_processing):

        # start a timer (to see how long processing and display takes)
        start_t = cv2.getTickCount()

        # if camera /video file successfully open then read frame
        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly
            if (ret == 0):
                keep_processing = False
                continue

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled
        # 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(
            frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)

        conf_count += len(confidences)
        conf_sum += np.sum(confidences)
        conf_average = conf_sum / conf_count
        writeScores(frame, conf_average, conf_count, total_count)
        
        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(frame,
                     classes[classIDs[detected_object]],
                     confidences[detected_object],
                     left,
                     top,
                     left + width,
                     top + height,
                     (255,
                      178,
                      50))

        # stop the timer and convert to ms. (to see how long processing takes
        stop_t = ((cv2.getTickCount() - start_t) /
                  cv2.getTickFrequency()) * 1000

        # Display efficiency information
        label = ('Inference time: %.2f ms' % stop_t) + \
            (' (Framerate: %.2f fps' % (1000 / stop_t)) + ')'
        cv2.putText(frame, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # to save in the video:
        frames.append(frame)

    # create the video:
    height, width, channel = frames[0].shape

    size = (width, height)

    print('Creating the result video!')
    out_video = cv2.VideoWriter(output_video_file_name, cv2.VideoWriter_fourcc(*'DIVX'), 3, size)

    for i in range(len(frames)):
        out_video.write(frames[i])
    out_video.release()

else:
    print("No video file specified.")

##########################################################################
