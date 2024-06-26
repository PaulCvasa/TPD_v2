import csv
import tkinter as tk
from tkinter import ttk
from threading import Thread
import time
import numpy as np
import psutil
from ultralytics import YOLO
import cv2
from CameraFrameGetter import CameraFrameGetter

def computeDistanceAndSendWarning(frame, results, roadType):
    boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding boxes in YOLOv8 format [x_center, y_center, width, height]
    classes = results[0].boxes.cls.int().cpu()  # Detected classes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    frameWidth = 1280
    frameHeight = 720

    # Define colors for different classes
    colors = {
        0: (0, 255, 0),  # Pedestrian - Green
        1: (255, 0, 0),  # Bicycle - Blue
        2: (0, 0, 255),  # Car - Red
        3: (255, 255, 0),  # Motorcycle - Yellow
        5: (255, 165, 0),  # Bus - Orange
        6: (255, 20, 147),  # Train - DeepPink
        7: (138, 43, 226)  # Truck - BlueViolet
    }

    for i, b in enumerate(boxes):
        classId = classes[i].item()
        x_center, y_center, width, height = boxes[i]
        if scores[i] > 0.4:  # if the confidence level is bigger than 40%
            # distance computed using detection box width, multiplied by 1000 to convert it in meters
            if width == 0:
                continue
            approxDist = round(1 / width, 5) * 1000
            # the power can be tweaked to affect granularity
            x_center_percentage = x_center / frame.shape[1] # percentage of how far the x coordinate is from the center of the frame

            # Draw bounding box
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            color = colors.get(classId, (255, 255, 255)) #default is black
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # if it's a pedestrian
            if classId == 0:
            #           image           text                                                 text position                                   text font         size       color   line width
                cv2.putText(frame, '{:0.1f} m'.format(approxDist / 3), (int(x_center_percentage * frameWidth + 20), int(x_center_percentage * frameHeight + 130)), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)  # show approximate distance
                if 0.2 < x_center_percentage < 0.8:  # if the object is in the ego vehicle path
                    cv2.putText(frame, '!WARNING!',(int(x_center_percentage * frameWidth), int(x_center_percentage * frameHeight)), cv2.FONT_ITALIC, 1.0,
                            (0, 0, 255), 3)

            # if it's a bicycle, car, a motorcycle, a bus, a train or a truck
            else:
                #           image         text                                                     text position                     text font         size       color   line width
                cv2.putText(frame, '{:0.1f} m'.format(approxDist), (int(x_center_percentage * frameWidth+20), int(x_center_percentage * frameHeight+130)), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)  # show approximate distance

                # check type of road, object in ego path and distance below threshold
                if roadType == 'highway' and 0.45 < x_center_percentage < 0.55 and approxDist <= 13:
                    cv2.putText(frame, '!WARNING!', (int(x_center_percentage * frameWidth), int(x_center_percentage * frameHeight)),
                                cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning

                elif roadType == 'normal' and 0.4 < x_center_percentage < 0.6 and approxDist <= 9:
                    cv2.putText(frame, '!WARNING!', (int(x_center_percentage * frameWidth), int(x_center_percentage * frameHeight)),
                                cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning

                elif roadType == 'city' and 0.3 < x_center_percentage < 0.7 and approxDist < 6:
                    cv2.putText(frame, '!WARNING!', (int(x_center_percentage * frameWidth), int(x_center_percentage * frameHeight)),
                                cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning


def detectLanes(frame):
    global lane_detection_counter, lane_detection_active
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # transform frame to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # add gaussian blur
    edges = cv2.Canny(blur, 50, 150)

    # Mask for the region of interest
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (400, (height // 2)+150),
        (880, (height // 2)+150),
        (width , height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Visualize the mask on the frame
    #mask_visualization = frame.copy()
    #cv2.polylines(mask_visualization, polygon, isClosed=True, color=(0, 255, 255), thickness=2)

    # Parameters: minLineLength increased for longer lines
    minLineLength = 100  # Set the minimum line length
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 50, np.array([]), minLineLength=minLineLength, maxLineGap=100)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if 0.5 < abs(slope) < 2 and length > minLineLength:  # Filter near-vertical lines and based on length
                if slope < 0:  # Left line
                    left_lines.append((length, line[0]))
                else:  # Right line
                    right_lines.append((length, line[0]))

    # Sort lines by length
    left_lines = sorted(left_lines, key=lambda x: x[0], reverse=True)
    right_lines = sorted(right_lines, key=lambda x: x[0], reverse=True)

    line_image = np.zeros_like(frame)

    # Draw the longest left and right lines
    left_detected = False
    right_detected = False

    if left_lines:
        x1, y1, x2, y2 = left_lines[0][1]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        left_detected = True

    if right_lines:
        x1, y1, x2, y2 = right_lines[0][1]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        right_detected = True

    # Update lane detection status
    if left_detected and right_detected:
        lane_detection_counter += 1
    else:
        lane_detection_counter = 0

    if lane_detection_counter >= 5:
        lane_detection_active = True
    else:
        lane_detection_active = False

    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) #mask_visualization

    # Display lane detection status
    if lane_detection_active:
        cv2.putText(combined_image, 'Lane Detection Active', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    return combined_image




# The detection method
def detection(input, roadType, filename, lane_detection_enabled):
    # Initialize the quantified and faster YOLOv8 model
    model = YOLO("yolov8n_integer_quant.tflite")#models/yolov8n_saved_model/yolov8n_float16.tflite

    #Results csv file
    results_file = 'detection_results.csv'

    # Variable for last frame processed time
    lastFrameTime = 0
    total_frames = 0

    with open(results_file, mode='a', newline='') as file:
        #writer = csv.writer(file)
        #writer.writerow(['Video Name', 'Frame', 'FPS', 'CPU Usage', 'Memory Usage', 'Preprocess Speed', 'Inference Speed', 'Postprocess Speed'])

        while True:
            frame = input.read()

            if cv2.waitKey(1) & 0xFF == ord('q') or input.stopped or frame is None:
                input.stop()
                cv2.destroyAllWindows()
                break

            if lane_detection_enabled:
                # Detect lanes and overlay on frame
                frame = detectLanes(frame)

            # Execute the prediction using YoloV8
            results = model.predict(source=frame, save=False, conf=0.5, iou=0.2, imgsz=640, half=True,
                                    save_txt=False, show=False, stream_buffer=True, augment=True, agnostic_nms=True,
                                    classes=[0, 1, 2, 3, 5, 6, 7])

            # Compute distances to TPs and send warnings
            computeDistanceAndSendWarning(frame, results, roadType)

            # Variable for current frame processed time
            currentFrameTime = time.time()
            fps = str(int(1 / (currentFrameTime - lastFrameTime)))
            lastFrameTime = currentFrameTime

            # Collect resource usage metrics
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent

            # Extract speed from YOLO results
            if len(results) > 0:
                preprocess_speed = results[0].speed['preprocess']
                inference_speed = results[0].speed['inference']
                postprocess_speed = results[0].speed['postprocess']

            # Write analytic data to csv
            #writer.writerow([filename, total_frames, fps, cpu_usage, memory_usage, preprocess_speed, inference_speed, postprocess_speed])

            total_frames += 1

            cv2.putText(frame, fps, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Traffic Participants Detection - Press Q to stop the detection', frame)


class TrafficParticipantsDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traffic Participants Detection - TPD")
        self.geometry("800x600")
        self.roadType = 'normal'
        self.lane_detection_enabled = False  # Initialize lane detection status
        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = ttk.Label(self, text="TRAFFIC PARTICIPANTS DETECTION - Raspberry PI 5 8GB", font=("Arial", 20))
        title_label.pack(pady=10)

        # Current road type
        self.road_type_label = ttk.Label(self, text="CURRENT ROAD TYPE SETTING: normal", font=("Arial", 14))
        self.road_type_label.pack(pady=10)

        # Road type buttons frame
        road_type_frame = ttk.Frame(self)
        road_type_frame.pack(pady=20)

        # Road type buttons
        city_button = ttk.Button(road_type_frame, text="CITY ROAD", command=lambda: self.set_road_type('city'), style='RoadType.TButton')
        city_button.grid(row=0, column=0, padx=10, pady=10)

        normal_button = ttk.Button(road_type_frame, text="NORMAL ROAD", command=lambda: self.set_road_type('normal'), style='RoadType.TButton')
        normal_button.grid(row=0, column=1, padx=10, pady=10)

        highway_button = ttk.Button(road_type_frame, text="HIGHWAY", command=lambda: self.set_road_type('highway'), style='RoadType.TButton')
        highway_button.grid(row=0, column=2, padx=10, pady=10)

        # Buttons frame
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(pady=20)

        # Start detection button
        start_button = ttk.Button(buttons_frame, text="START DETECTION", command=self.start_detection, style='Start.TButton')
        start_button.grid(row=0, column=0, padx=10, pady=10)

        # Test detection button
        test_button = ttk.Button(buttons_frame, text="TEST DETECTION", command=self.test_detection, style='Test.TButton')
        test_button.grid(row=0, column=1, padx=10, pady=10)

        # Exit button
        exit_button = ttk.Button(buttons_frame, text="EXIT", command=self.quit, style='Exit.TButton')
        exit_button.grid(row=0, column=2, padx=10, pady=10)

        # Author label
        author_label = tk.Label(self, text="Author: Paul Cvasa", font=("Arial", 10))
        author_label.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

        # Version Label
        version_label = tk.Label(self, text="version: 2.0", font=("Arial", 10))
        version_label.place(relx=0.0, rely=1.0, anchor='sw', x=10, y=-10)

        # Lane detection status label
        self.lane_detection_label = ttk.Label(self, text="Lane Detection: Disabled", font=("Arial", 14))
        self.lane_detection_label.pack(pady=10)

        # Lane detection toggle button
        self.toggle_lane_detection_button = ttk.Button(self, text="Enable Lane Detection",
                                                       command=self.toggle_lane_detection, style='RoadType.TButton')
        self.toggle_lane_detection_button.pack(pady=10)


        # Custom styles for buttons
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 12), padding=10)
        style.configure('Start.TButton', background='green', foreground='white')
        style.map('Start.TButton', background=[('active', 'dark green')])

        style.configure('Test.TButton', background='yellow', foreground='black')
        style.map('Test.TButton', background=[('active', 'gold')])

        style.configure('Exit.TButton', background='red', foreground='white')
        style.map('Exit.TButton', background=[('active', 'dark red')])

        style.configure('RoadType.TButton', background='slate gray', foreground='black')
        style.map('RoadType.TButton', background=[('active', 'slate gray')])

    def set_road_type(self, road_type):
        self.roadType = road_type
        self.road_type_label.config(text=f"CURRENT ROAD TYPE SETTING: {road_type}")

    def start_detection(self):
        print(self.roadType)
        videoInput = CameraFrameGetter(0, 480, 480).start()
        detection(videoInput, self.roadType, filename='realtime-camera', lane_detection_enabled=self.lane_detection_enabled)

    def test_detection(self):
        print(self.roadType)
        filename = 'testRecs/city_Trim.mp4'
        testInput = CameraFrameGetter(filename, 'testVideo', 480, 240).start()
        detection(testInput, self.roadType, filename, lane_detection_enabled=self.lane_detection_enabled)

    def toggle_lane_detection(self):
        self.lane_detection_enabled = not self.lane_detection_enabled
        if self.lane_detection_enabled:
            self.lane_detection_label.config(text="Lane Detection: Enabled")
            self.toggle_lane_detection_button.config(text="Disable Lane Detection")
        else:
            self.lane_detection_label.config(text="Lane Detection: Disabled")
            self.toggle_lane_detection_button.config(text="Enable Lane Detection")


if __name__ == "__main__":
    app = TrafficParticipantsDetectionApp()
    app.mainloop()