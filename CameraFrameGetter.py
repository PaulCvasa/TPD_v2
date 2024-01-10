import time
from threading import Thread
import cv2


class CameraFrameGetter:
    def __init__(self, src=0, detectionType='camera'):
        self.detectionType = detectionType
        # initialize the camera stream and read the first frame from input
        self.input = cv2.VideoCapture(src)
        (self.success, self.frame) = self.input.read()
        # flag for thread status
        self.stopped = False

    def start(self):
        # thread is created and started reading frames
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        fps = self.input.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / fps  # Time (in seconds) between frames
        # this keeps the thread alive
        while True:
            # if a flag is triggered, kill the thread
            if self.stopped or not self.success:
                return
            # continue reading from input
            (self.success, self.frame) = self.input.read()
            if self.detectionType == 'testVideo':
                time.sleep(frame_time)

    def read(self):
        # returns the last frame read
        return self.frame

    def stop(self):
        # changes the thread's status flag to stopped
        self.stopped = True
