import threading
import time

from ultralytics import YOLO
import cv2
import PySimpleGUI as psg
from CameraFrameGetter import *


# def computeDistanceAndSendWarning(frame, boxes, classes, scores, roadType):
#     for i, b in enumerate(boxes[0]):  # iterate through detection boxes
#         if scores[0][i] > 0.4:  # if the confidence level is bigger than 50%
#             # we subtract the width difference from 1, so the value will be towards 0 when the object is closer and to 1 when it's far away
#             approxDist = round((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 8, 2) * 20  # distance computed using detection box width, multiplied by 20 to convert in meters
#                                                                                      # the power can be tweaked to affect granularity
#             middleX = (boxes[0][i][1] + boxes[0][i][3]) / 2  # middle of the X coord of detected object
#             middleY = (boxes[0][i][0] + boxes[0][i][2]) / 2  # middle of the Y coord of detected object
#
#             # if it's a bicycle,       a car,              a motorcycle,           a bus,              a train                 or a truck
#             if classes[0][i] == 2 or classes[0][i] == 3 or classes[0][i] == 4 or classes[0][i] == 6 or classes[0][i] == 7 or classes[0][i] == 8:
#                 #           image         text                          text position                     text font            size       color   line width
#                 cv2.putText(frame, '{:0.1f} m'.format(approxDist), (int(middleX * 1300), int(middleY * 550)), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)  # show approximate distance
#                 if roadType == 'highway':      # verify type of road
#                     if 0.45 < middleX < 0.55:    # if the object is in the ego vehicle path
#                         if approxDist <= 13:  # if the approximate distance is smaller than the threshold
#                             cv2.putText(frame, '!WARNING!', (int(middleX * 1280), int(middleY * 720)), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning
#                 if roadType == 'normal':       # verify type of road
#                     if 0.4 < middleX < 0.6:      # if the object is in the ego vehicle path
#                         if approxDist <= 9:  # if the approximate distance is smaller than the threshold
#                             cv2.putText(frame, '!WARNING!', (int(middleX * 1280), int(middleY * 720)), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning
#                 if roadType == 'city':         # verify type of road
#                     if 0.3 < middleX < 0.7:      # if the object is in the ego vehicle path
#                         if approxDist < 6:   # if the approximate distance is smaller than the threshold
#                             cv2.putText(frame, '!WARNING!', (int(middleX * 1280), int(middleY * 720)), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning
#             elif classes[0][i] == 1 :  # if it's a pedestrian
#                 #           image           text                                  text position                 text font         size       color   line width
#                 cv2.putText(frame, '{:0.1f} m'.format(approxDist / 3), (int(middleX * 1300), int(middleY * 550)), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)  # show approximate distance
#                 if 0.2 < middleX < 0.8:  # if the object is in the ego vehicle path
#                     cv2.putText(frame, '!WARNING!', (int(middleX * 1280) , int(middleY * 720)), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)

def computeDistanceAndSendWarning(frame, results, roadType):
    boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding boxes in YOLOv8 format [x_center, y_center, width, height]
    classes = results[0].boxes.cls.int().cpu()  # Detected classes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    for i, b in enumerate(boxes):
        x_center, y_center, width, height = boxes[i]
        if scores[i] > 0.4:  # if the confidence level is bigger than 40%
            # we subtract the width difference from 1, so the value will be towards 0 when the object is closer and to 1 when it's far away
            #approxDist = round((1 - width / frame.shape[1]) ** 8, 2) * 20  # distance computed using detection box width, multiplied by 20 to convert in meters
            approxDist = round(1 / width, 4) * 1000
            # the power can be tweaked to affect granularity
            x_center_percentage = x_center / frame.shape[1] # percentage of how far the x coordinate is from the center of the frame
            # if it's a bicycle,    a car,            a motorcycle,       a bus,              a train             or a truck
            if classes[i] == 1 or classes[i] == 2 or classes[i] == 3 or classes[i] == 5 or classes[i] == 6 or classes[i] == 7:
                #           image         text                                                     text position                     text font         size       color   line width
                cv2.putText(frame, '{:0.1f} m'.format(approxDist), (int(x_center_percentage * 1300), int(x_center_percentage * 550)), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)  # show approximate distance
                if roadType == 'highway':  # verify type of road
                    if 0.45 < x_center_percentage < 0.55:  # if the object is in the ego vehicle path
                        if approxDist <= 13:  # if the approximate distance is smaller than the threshold
                            cv2.putText(frame, '!WARNING!', (int(x_center_percentage * 1280), int(x_center_percentage * 720)),
                                        cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning
                if roadType == 'normal':  # verify type of road
                    if 0.4 < x_center_percentage < 0.6:  # if the object is in the ego vehicle path
                        if approxDist <= 9:  # if the approximate distance is smaller than the threshold
                            cv2.putText(frame, '!WARNING!', (int(x_center_percentage * 1280), int(x_center_percentage * 720)),
                                        cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning
                if roadType == 'city':  # verify type of road
                    if 0.3 < x_center_percentage < 0.7:  # if the object is in the ego vehicle path
                        if approxDist < 6:  # if the approximate distance is smaller than the threshold
                            cv2.putText(frame, '!WARNING!', (int(x_center_percentage * 1280), int(x_center_percentage * 720)),
                                        cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning
            elif classes[i] == 0:  # if it's a pedestrian
                #           image           text                                                 text position                           text font         size       color   line width
                cv2.putText(frame, '{:0.1f} m'.format(approxDist / 3), (int(x_center_percentage * 1300), int(x_center_percentage * 550)), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)  # show approximate distance
                if 0.2 < x_center_percentage < 0.8:  # if the object is in the ego vehicle path
                    cv2.putText(frame, '!WARNING!', (int(x_center_percentage * 1280), int(x_center_percentage * 720)), cv2.FONT_ITALIC, 1.0,
                                (0, 0, 255), 3)



# Different method used for the webcam input, because of the CameraFrameGetter optimization
def detection(input, roadType):
    # Initialize model
    model = YOLO("yolov8n.pt")


    # Variable for last frame processed time
    lastFrameTime = 0
    # Variable for current frame processed time
    currentFrameTime = 0

    while True:
        frame = input.read()

        # Execute the prediction using YoloV8
        results = model.predict(source=frame, save=False, conf=0.5, save_txt=False, show=True)

        #for r in results:
            #detection_boxes = r.boxes
            #classifications
        #detection_boxes = results[0].boxes.xyxy
        #classifications = results.names[results.xyxy[0][:, 5].cpu().numpy().astype(int)]
        #scores = results.xyxy[0][:, 4].cpu().numpy()

        # Compute distances to TPs and send warnings
        t = threading.Thread(target=computeDistanceAndSendWarning, args=[frame, results, roadType])
        t.start()
        t.join()

        currentFrameTime = time.time()
        fps = str(int(1 / (currentFrameTime - lastFrameTime)))
        lastFrameTime = currentFrameTime

        cv2.putText(frame, fps, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Traffic Participants Detection - Press Q to stop the detection',
                    cv2.resize(frame, (950, 600)))

        if cv2.waitKey(25) & 0xFF == ord('q') or input.stopped:
            input.stop()
            cv2.destroyAllWindows()
            break


def main():
    roadType = 'normal'

    psg.theme("Dark")
    mainFont = "Comic Sans MS", "10"
    secondaryFont = "Comic Sans MS", "14"
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
        [psg.Text("TRAFFIC PARTICIPANTS DETECTION", size=(120,0), justification="center", font=("Comic Sans MS", "20"))],
        [psg.Text("CURRENT ROAD TYPE SETTING: ", size=(30, 25), font=secondaryFont), psg.Text("normal", size=(60, 25), key='setting', font=secondaryFont)],
        [psg.Column(mainButtonsLine, vertical_alignment='center', justification='center', k='-C-')],
        [psg.Column(settingsButtonsLine, vertical_alignment='center', justification='center', k='-C-')]
    ]

    mainWindow = psg.Window("TRAFFIC PARTICIPANTS DETECTION", mainLayout, resizable=True, finalize=True)
    mainWindow.Maximize()
    while True:
        event, values = mainWindow.read(timeout=20)
        if event == "EXIT" or event == psg.WIN_CLOSED:
            break
        elif event == "START DETECTION":
            print(roadType)
            videoInput = CameraFrameGetter(0).start()
            detection(videoInput, roadType)
        elif event == "TEST DETECTION":
            print(roadType)
            #testInput = cv2.VideoCapture('testRecs/city2.mp4')
            testInput = CameraFrameGetter('testRecs/city2.mp4', 'testVideo').start()
            detection(testInput, roadType)
        elif event == "city":
            roadType = "city"
            mainWindow['setting'].update(value=roadType)
            mainWindow["city"].Update(button_color=('Black'))
            mainWindow["normal"].Update(button_color=('Grey'))
            mainWindow["highway"].Update(button_color=('Grey'))
        elif event == "normal":
            roadType = "normal"
            mainWindow['setting'].update(value=roadType)
            mainWindow["normal"].Update(button_color=('Black'))
            mainWindow["city"].Update(button_color=('Grey'))
            mainWindow["highway"].Update(button_color=('Grey'))
        elif event == "highway":
            roadType = "highway"
            mainWindow['setting'].update(value=roadType)
            mainWindow["highway"].Update(button_color=('Black'))
            mainWindow["normal"].Update(button_color=('Grey'))
            mainWindow["city"].Update(button_color=('Grey'))


main()
