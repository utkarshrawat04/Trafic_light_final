from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

# Create video capture objects for each video source
cap = cv2.VideoCapture("../Videos/cars.mp4")
cap1 = cv2.VideoCapture("../Videos/q.mp4")
cap2 = cv2.VideoCapture("../Videos/oo.mp4")
cap3 = cv2.VideoCapture("../Videos/l.mp4")

# Initialize YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")

# The YOLO labels
labels = ["Human", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck", "Boat",
          "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat",
          "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella",
          "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
          "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup",
          "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli",
          "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Sofa", "Pottedplant", "Bed",
          "Diningtable", "Toilet", "TVmonitor", "Laptop", "Mouse", "Remote", "Keyboard", "Phone",
          "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors",
          "Teddy Bear", "Hair Dryer", "Toothbrush"]

# Initialize SORT tracker for each video
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker1 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker3 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define lines for each video
line = [400, 297, 673, 297]
line1 = [100, 400, 700, 400]
line2 = [50, 250, 300, 250]
line3 = [500, 400, 1300, 400]


# Initialize counts for each video
total_count = []
total_count1 = []
total_count2 = []
total_count3 = []

while True:
    # Reads frames from each video source
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    if not ret:
        break  # Exit the loop when the video ends

    # Process frames using YOLO for each video
    results = model(frame, stream=True)
    results1 = model(frame1, stream=True)
    results2 = model(frame2, stream=True)
    results3 = model(frame3, stream=True)


    #creates an empty array
    detections = np.empty((0, 5))
    detections1 = np.empty((0, 5))
    detections2 = np.empty((0, 5))
    detections3 = np.empty((0, 5))

    # Process results for the first video
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0] #size of the box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1


#calculates confidence for each vehical
            conf = math.ceil((box.conf[0] * 100)) / 100
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))
            cls = int(box.cls[0])
            currentClass = labels[cls]

            if currentClass in ["Car", "Motorbike", "Bus", "Truck"] and conf > 0.3:
                cvzone.putTextRect(frame, f'{labels[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.5, thickness=1,
                                  offset=5)

    # Process results for the second video (frame1)
    for r in results1:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections1 = np.vstack((detections1, currentArray))
            cls = int(box.cls[0])
            currentClass = labels[cls]

            if currentClass in ["Car", "Motorbike", "Bus", "Truck"] and conf > 0.3:
                cvzone.putTextRect(frame1, f'{labels[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.5, thickness=1,offset=5)

    # Process results for the third video (frame2)
    for r in results2:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections2 = np.vstack((detections2, currentArray))
            cls = int(box.cls[0])
            currentClass = labels[cls]

            if currentClass in ["Car", "Motorbike", "Bus", "Truck"] and conf > 0.3:
                cvzone.putTextRect(frame2, f'{labels[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.5, thickness=1,offset=5)

    # Process results for the fourth video (frame3)
    for r in results3:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections3 = np.vstack((detections3, currentArray))
            cls = int(box.cls[0])
            currentClass = labels[cls]

            if currentClass in ["Car", "Motorbike", "Bus", "Truck"] and conf > 0.3:
                cvzone.putTextRect(frame3, f'{labels[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.5, thickness=1,offset=5)

    # Update object tracking for each video
    resultsTracker = tracker.update(detections)
    resultsTracker1 = tracker1.update(detections1)
    resultsTracker2 = tracker2.update(detections2)
    resultsTracker3 = tracker3.update(detections3)

#red circle
    cv2.circle(frame, (60, 120), 50, (0, 0, 225), cv2.FILLED)
    cv2.circle(frame1, (60, 120), 50, (0, 0, 225), cv2.FILLED)
    cv2.circle(frame2, (60, 120), 50, (0, 0, 225), cv2.FILLED)
    cv2.circle(frame3, (60, 120), 50, (0, 0, 225), cv2.FILLED)

    # Count objects and highlight the frame with the largest count
    count1 = len(total_count)
    count2 = len(total_count1)
    count3 = len(total_count2)
    count4 = len(total_count3)

    # Compare counts and highlight the frame with the largest count
    largest_count = max(count1, count2, count3, count4)

    # Display the frame with the largest count
    if largest_count == count1:
        cv2.circle(frame, (60, 120), 50, (0, 225, 0), cv2.FILLED)
        largest_frame = frame
    elif largest_count == count2:
        cv2.circle(frame1, (60, 120), 50, (0, 225, 0), cv2.FILLED)
        largest_frame = frame1
    elif largest_count == count3:
        cv2.circle(frame2, (60, 120), 50, (0, 225, 0), cv2.FILLED)
        largest_frame = frame2
    elif largest_count == count4:
        cv2.circle(frame3, (60, 120), 50, (0, 225, 0), cv2.FILLED)
        largest_frame = frame3


    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5)
    cv2.line(frame1, (line1[0], line1[1]), (line1[2], line1[3]), (0, 0, 255), 5)
    cv2.line(frame2, (line2[0], line2[1]), (line2[2], line2[3]), (0, 0, 255), 5)
    cv2.line(frame3, (line3[0], line3[1]), (line3[2], line3[3]), (0, 0, 255), 5)


    # Update the frames with bounding boxes and counts for the first video
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if line[0] < cx < line[2] and line[1] - 15 < cy < line[1] + 15:
            if id not in total_count:
                total_count.append(id)
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 225, 0), 5)

        # Display the count on the screen
        cvzone.putTextRect(frame, f' Count : {len(total_count)}', (40, 50))

    # Update the frames with bounding boxes and counts for the second video (frame1)
    for result in resultsTracker1:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame1, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame1, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if line1[0] < cx < line1[2] and line1[1] - 15 < cy < line1[1] + 15:
            if id not in total_count1:
                total_count1.append(id)
                cv2.line(frame1, (line1[0], line1[1]), (line1[2], line1[3]), (0, 225, 0), 5)

        # Display the count on the screen
        cvzone.putTextRect(frame1, f' Count : {len(total_count1)}', (40, 50))

    # Update the frames with bounding boxes and counts for the third video (frame2)
    for result in resultsTracker2:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame2, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame2, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if line2[0] < cx < line2[2] and line2[1] - 15 < cy < line2[1] + 15:
            if id not in total_count2:
                total_count2.append(id)
                cv2.line(frame2, (line2[0], line2[1]), (line2[2], line2[3]), (0, 225, 0), 5)

        # Display the count on the screen
        cvzone.putTextRect(frame2, f' Count : {len(total_count2)}', (40, 50))

    # Update the frames with bounding boxes and counts for the fourth video (frame3)
    for result in resultsTracker3:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame3, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame3, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if line3[0] < cx < line3[2] and line3[1] - 15 < cy < line3[1] + 15:
            if id not in total_count3:
                total_count3.append(id)
                cv2.line(frame3, (line3[0], line3[1]), (line3[2], line3[3]), (0, 225, 0), 5)

        # Display the count on the screen
        cvzone.putTextRect(frame3, f' Count : {len(total_count3)}', (40, 50))

    # Show the frames for each video in separate windows
    cv2.namedWindow('Cam 1', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Cam 2', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Cam 3', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Cam 4', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    cv2.imshow('Cam 1', frame)
    cv2.imshow('Cam 2', frame1)
    cv2.imshow('Cam 3', frame2)
    cv2.imshow('Cam 4', frame3)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture objects and close all windows
cap.release()
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()






