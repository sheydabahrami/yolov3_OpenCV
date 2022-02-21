# Import required libraries
import tensorflow as tf
import cv2
import numpy as np
import base64
from flask import Flask, jsonify, request, render_template, Response, send_file
from PIL import Image
import io
from io import BytesIO


# define this is a flask app
app = Flask(__name__)


@app.route('/detection', methods=['POST'])
def detect():
# Load the COCO class labels our YOLO model was trained on. (80 labels)
    LABELS = open("yolov3.txt").read().strip().split("\n")
        
    # Initialize list of colors to represent class labels, use bright colors so its easier to read
    COLORS = np.random.randint(0, 100, size=(len(LABELS), 3), dtype="uint8")

    # Load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

    # Load our input image and grab dimensions
    filestr = request.files['img'].read()
    npimg = np.frombuffer(filestr, np.uint8)
    # convert numpy array to image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    (H, W) = image.shape[:2]
    
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()

    classes = [] 
    classes = [line.strip() for line in ln] 
    outputlayers = net.getUnconnectedOutLayersNames()  

    # Construct a blob from the input image and perform forward pass of YOLO to get bounding boxes and probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(outputlayers) 

    # Initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # Filter out weak predictions by ensuring detected probability is greater some threshold
            if confidence > 0.5:
                # Scale bounding box coordinates back relative to size of image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive top left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maximum suppression to suppress weak overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Loop over the indexes and draw bounding boxes
    for i in idxs.flatten():
        # Extract bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
    
        # Draw a bounding box using opencv rectangle and label the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # show the output image
    retimage = Image.fromarray(image)

    #retimage.save("C://tempoutput//1.jpg", format="JPEG", quality=100)

    buffer = io.BytesIO()
    retimage.save(buffer, format="JPEG", quality=100)
    return Response(buffer.getvalue(), mimetype='image/jpeg')


if __name__=="__main__":
    app.run(debug=True)




    
