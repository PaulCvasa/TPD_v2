import threading
import time
import numpy as np
from ultralytics import YOLO
import cv2
import PySimpleGUI as psg
from CameraFrameGetter import CameraFrameGetter

def computeDistanceAndSendWarning(frame, results, roadType):
    boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding boxes in YOLOv8 format [x_center, y_center, width, height]
    classes = results[0].boxes.cls.int().cpu()  # Detected classes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    frameWidth = 1280
    frameHeight = 720

    for i, b in enumerate(boxes):
        x_center, y_center, width, height = boxes[i]
        if scores[i] > 0.4:  # if the confidence level is bigger than 40%
            # distance computed using detection box width, multiplied by 1000 to convert it in meters
            if width == 0:
                continue
            approxDist = round(1 / width, 5) * 1000
            # the power can be tweaked to affect granularity
            x_center_percentage = x_center / frame.shape[1] # percentage of how far the x coordinate is from the center of the frame

            # if it's a pedestrian
            if classes[i] == 0:
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
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Mask for bottom half of the image
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, height // 2),
        (0, height // 2)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=100)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if 0.5 < abs(slope) < 2 and length > 50:  # Filter near-vertical lines and based on length
                if slope < 0:  # Left line
                    left_lines.append((length, line[0]))
                else:  # Right line
                    right_lines.append((length, line[0]))

    # Sort lines by length
    left_lines = sorted(left_lines, key=lambda x: x[0], reverse=True)
    right_lines = sorted(right_lines, key=lambda x: x[0], reverse=True)

    line_image = np.zeros_like(frame)

    # Draw the longest left and right lines
    if left_lines:
        x1, y1, x2, y2 = left_lines[0][1]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    if right_lines:
        x1, y1, x2, y2 = right_lines[0][1]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined_image


# The detection method
def detection(input, roadType):
    # Initialize the quantified and faster YOLOv8 model
    model = YOLO("yolov8n_integer_quant.tflite")#models/yolov8n_saved_model/yolov8n_float16.tflite

    # Variable for last frame processed time
    lastFrameTime = 0

    while True:
        frame = input.read()

        if frame is None:
            print('frame skippedq')
            continue

        # Detect lanes and overlay on frame
        frame_with_lanes = detectLanes(frame)

        # Execute the prediction using YoloV8
        results = model.predict(source=frame_with_lanes, save=False, conf=0.5, iou=0.2, imgsz=640, half=True, save_txt=False, show=True, stream_buffer=True, augment=True, agnostic_nms=True,
                                classes=[0, 1, 2, 3, 5, 6, 7])

        # Compute distances to TPs and send warnings
        computeDistanceAndSendWarning(frame_with_lanes, results, roadType)

        # Variable for current frame processed time
        currentFrameTime = time.time()
        fps = str(int(1 / (currentFrameTime - lastFrameTime)))
        lastFrameTime = currentFrameTime

        cv2.putText(frame_with_lanes, fps, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Traffic Participants Detection - Press Q to stop the detection', frame_with_lanes)

        if cv2.waitKey(1) & 0xFF == ord('q') or input.stopped:
            input.stop()
            cv2.destroyAllWindows()
            break


def main():
    roadType = 'normal'

    psg.theme("Dark")
    mainFont = "Arial", "10"
    secondaryFont = "Arial", "14"
    # Main buttons layout
    mainButtonsLine = [
        [psg.Button("START DETECTION", size=(20, 7), font=mainFont),
         psg.Button("TEST DETECTION", size=(20, 7), button_color='Blue', font=mainFont),
         psg.Button("EXIT", size=(20, 7), button_color='Red', font=mainFont)]
    ]

    # Settings Buttons layout
    settingsButtonsLine = [
        [psg.Button("CITY ROAD", size=(20, 7), button_color='Gray', font=mainFont, key="city"),
         psg.Button("NORMAL ROAD", size=(20, 7), button_color='Black', font=mainFont, key="normal"),
         psg.Button("HIGHWAY", size=(20, 7), button_color='Gray', font=mainFont, key="highway")]
    ]

    # Main layout window
    mainLayout = [
        [psg.Text("TRAFFIC PARTICIPANTS DETECTION", size=(120,0), justification="center", font=("Arial", "20"))],
        [psg.Text("CURRENT ROAD TYPE SETTING: ", size=(30, 25), font=secondaryFont), psg.Text("normal", size=(60, 25), key='setting', font=secondaryFont)],
        [psg.Column(mainButtonsLine, vertical_alignment='center', justification='center', k='-C-')],
        [psg.Column(settingsButtonsLine, vertical_alignment='center', justification='center', k='-C-')]
    ]

    mainWindow = psg.Window("TRAFFIC PARTICIPANTS DETECTION", mainLayout, resizable=True, finalize=True)
    #mainWindow.Maximize()
    while True:
        event, values = mainWindow.read(timeout=20)
        if event == "EXIT" or event == psg.WIN_CLOSED:
            break
        elif event == "START DETECTION":
            print(roadType)
            videoInput = CameraFrameGetter(0, 480,480).start()
            detection(videoInput, roadType)
        elif event == "TEST DETECTION":
            print(roadType)
            #testInput = cv2.VideoCapture('testRecs/city2.mp4')
            testInput = CameraFrameGetter('city2.mp4', 'testVideo', 480, 240).start()
            detection(testInput, roadType)
        elif event in ["city", "normal", "highway"]:
            roadType = event
            mainWindow['setting'].update(value=roadType)
            for key in ["city", "normal", "highway"]:
                mainWindow[key].Update(button_color=('Black' if key == event else 'Gray'))


main()
