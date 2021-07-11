import numpy as np
import time
import cv2
import math as m
import matplotlib.image as mpimg
import math
framewidth = 0
frameheight = 0
list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "person"]

values = {
    "two_"
}
two_wheeler = ["bicycle","motorbike"]
four_wheeler = ["car", "bus"]
pedestrian = ["person"]

preDefinedConfidence = 0.5
preDefinedThreshold = 0.3

configPath = ".\\yolo-coco\\yolov3.cfg"
weightsPath = ".\\yolo-coco\\yolov3.weights"

# coco.names (string labels) from yolo
LABELS = open('.\\yolo-coco\\coco.names').read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

previous_frame_detections = []


class BoundingBox:
    def __init__(self, x, y, id, t):
        self.x = x
        self.y = y
        self.id = id
        self.t = t


flag = 0
threshold = 20  #m.inf


def compare(old, new):
    global flag
    global threshold
    x_old, y_old = old.x, old.y
    x_new, y_new = new.x, new.y
    dist = m.sqrt((x_old - x_new)**2 + (y_old-y_new)**2)
    if(dist <= min(old.t, new.t)):
        return True
    return False


def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    # ensure at least one detection exists
    cv2.rectangle(frame, (0, int(5/8*frameheight)), (framewidth, int(3/4*frameheight)), (255,0,0), 2)
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w//2)
            centerY = y + (h//2)

            # if(validcomparison(centerX, centerY) == True):
                # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                        confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w//2), y + (h//2)),
                        2, (0, 0xFF, 0), thickness=2)


def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, id, t = current_box
    # Iterating through all the k-dimensional trees
    if(len(previous_frame_detections) == 0):
        return False
    for i in range(len(previous_frame_detections)):
        oldBoundingBox = previous_frame_detections[i]
        newBoundingBox = BoundingBox(centerX, centerY, id, t)
        if(compare(oldBoundingBox, newBoundingBox) == True):
            current_detections[(centerX, centerY,t)] = previous_frame_detections[i].id
            return True
    return False


def displayVehicleCount(frame, vehicle_count):
    cv2.putText(
        frame,  # Image
        'Detected Vehicles: ' + str(vehicle_count),  # Label
        (20, 20),  # Position
        cv2.FONT_HERSHEY_SIMPLEX,  # Font
        0.8,  # Size
        (0, 0xFF, 0),  # Color
        2,  # Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )


def validcomparison(centerX, centerY):
    height_s = frameheight * 5/8
    height_e = frameheight * 3/4    
    if(height_s <= centerY and centerY <= height_e):
        return True
    return False


def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w//2)
            centerY = y + (h//2)


# When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if (LABELS[classIDs[i]] in list_of_vehicles and validcomparison(centerX, centerY) == True):
                t = max(w//2, h//2)
                current_detections[(centerX, centerY, t)] = vehicle_count
                if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, vehicle_count, t), current_detections)):
                    vehicle_count += 1

                ID = current_detections.get((centerX, centerY,t))
                # If there are two detections having the same ID due to being too close,
                # then assign a new ID to current detection.
                if (list(current_detections.values()).count(ID) > 1):
                    current_detections[(centerX, centerY)] = vehicle_count
                    vehicle_count += 1

                # Display the ID at the center of the box
                cv2.putText(frame, str(ID), (centerX, centerY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    return vehicle_count, current_detections


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
USE_GPU = True

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
if USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

num_frame = 41
vehicle_count = 0
for k in range(1, 160):
    print("new frame", k+1)
    frame = cv2.imread('.\\images\\ezgif-frame-0'+str(k)+'.jpg')
    inputWidth = frame.shape[1]
    inputHeight = frame.shape[0]


    boxes, confidences, classIDs = [], [], []
    frame = frame[inputHeight//2:inputHeight+1, 0:inputWidth//2]

    frameheight = int((inputHeight//2+1)//32) *32
    framewidth = int((inputWidth//2)//32) *32

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (framewidth, frameheight),swapRB=True, crop=False)

    net.setInput(blob)
    # start = time.time()
    layerOutputs = net.forward(ln)
    # end = time.time()
    for output in layerOutputs:
        # loop over each of the detections
        for i, detection in enumerate(output):
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > preDefinedConfidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * \
                    np.array([framewidth, frameheight,framewidth, frameheight])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
                            preDefinedThreshold)

    drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

    vehicle_count, current_detections = count_vehicles(
        idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)

    displayVehicleCount(frame, vehicle_count)

    # image saved to disk (name: saved_image, src: frame)
    path = 'C:\\Users\\ayush\\Desktop\\saved\\saved_image-0'+str(k)+'.jpg'
    cv2.imwrite(path, frame)

# Updating with the current frame detections

    previous_frame_detections = []
    for cx, cy,t in current_detections:
        previous_frame_detections.append(BoundingBox(cx, cy, current_detections.get((cx, cy, t)), t))