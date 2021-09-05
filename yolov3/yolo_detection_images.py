import numpy as np
import cv2
import json

def annotateImages():
    confidenceThreshold = 0.8 #confidence threshold of boxes to pick up
    NMSThreshold = 0.3

    modelConfiguration = 'cfg/yolov3.cfg'
    modelWeights = 'yolov3.weights'

    labelsPath = 'coco.names'
    labels = open(labelsPath).read().strip().split('\n')

    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    image = cv2.imread('images/dog.jpg')
    (H, W) = image.shape[:2]

    #Determine output layer names
    layerName = net.getLayerNames()
    layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

    #for bounding box json output
    outputs = {}

    #ensure detections exist
    if len(detectionNMS) > 0:
        outputs["image"] = {}
        outputs["image"]["annotations"] = []
        for i in detectionNMS.flatten():
            detection = {}
            detection["label"] = labels[classIDs[i]]
            detection["cf"] = confidences[i]
            detection["x"] = boxes[i][0]
            detection["y"] = boxes[i][1]
            detection["width"] = boxes[i][2]
            detection["height"] = boxes[i][3]
            outputs["image"]["annotations"].append(detection)
            #print(labels[classIDs[i]]) #label for class id
            #print(confidences[i]) #confidence rating
            #print(boxes[i][0]) #x coordinate (top left)
            #print(boxes[i][1]) #y coordinate (top left)
            #print(boxes[i][2]) #width
            #print(boxes[i][3]) #height
    else:
        outputs["image"] = "No detections found"
    return json.dumps(outputs)