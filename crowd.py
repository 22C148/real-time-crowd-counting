import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Ensure correct paths
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Fixed indexing

# Start video capture
cap = cv2.VideoCapture(0)  # Use webcam or replace with video path

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    people_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            people_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add count text
    cv2.putText(frame, f"People Count: {people_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert BGR to RGB for matplotlib
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display frame with matplotlib
    plt.imshow(rgb_frame)
    plt.axis("off")
    plt.pause(0.001)  # Pause to update the plot

    # Press 'q' to exit
    if plt.waitforbuttonpress(0.001):  # Wait for a key press
        break

cap.release()
plt.close()
