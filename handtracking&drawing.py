import os
os.environ["OPENCV_VIDEOIO_MSMF-ENABLE_HW_TRANSFORMS"]= '0'

import cv2
cv2.useOptimized()
import mediapipe as mp
import numpy as np
import time


from cvzone.HandTrackingModule import HandDetector
from threading import Thread

class WebcamStream:
    """WebcamStream captures frames from the webcam. """

    def __init__(self, stream_id= 0):
        self.stream_id = stream_id
        self.vcap= cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream. ")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("Fps of input stream:{}".format(fps_input_stream))

        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Esiting] No more frames to read')
            exit(0)
        self.stopped= True
        #thread instantiation
        self.t = Thread(target=self.update, args=())
        self.t.daemon= True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    def read(self):
        return self.frame
    def stop(self):
        self.stopped= True

class HandGestureDetector:
    """HandGestureDetector detects hand and draws on the x and y co-ordinates of the index finger """

    def __init__(self, cap, hands, mpHands, detector, canvas):
        self.mpHands= mpHands
        self.hands= hands
        self.cap= cap
        self.detector = detector
        self.canvas= canvas
        self.x= 0
        self.y= 0
        self.drawing= False
        self.tool= None
        self.thickness= 3
        self.input_size= (1200, 1080)
        self.pTime= 0
        self.cTime= 0

    def run_detection(self):
        num_frames_processed= 0
        start = time.time()
        while True:
            img= self.cap.read()
            img_resized= cv2.resize(img, self.input_size)
            hand, img= self.detector.findHands(img_resized)
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results= self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_finger_tip= hand_landmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP]
                    cx, cy = int(index_finger_tip.x * self.canvas.shape[1]),  int(index_finger_tip.y * self.canvas.shape[0])

                    if hand:
                        handType = hand[0]['type']
                        if handType == 'Left':
                            self.drawing = False
                            cv2.circle(self.canvas, (cx, cy), 150, (255, 255, 255), thickness=150)
                        elif handType == "Right":
                            if self.drawing:
                                cv2.line(self.canvas, (self.x, self.y),(cx, cy), 0, thickness= 10, )
                                self.x = cx
                                self.y = cy
                            self.drawing = True
            self.cTime = time.time()
            fps = 1/(self.cTime - self.pTime)
            self.pTime = self.cTime

            cv2.imshow("Canvas", self.canvas)
            cv2.putText(img, f'Fps:{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("image", img)

            if cv2.waitKey(5) &0xFF == 27:
                break
        end= time.time()
        self.cap.stop()

    def __del__(self):
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = WebcamStream(stream_id= 0)
    cap.start()
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode= False,
                          max_num_hands=2,
                          min_detection_confidence= 0.5,
                          min_tracking_confidence= 0.5)
    detector= HandDetector(detectionCon=0.8, maxHands= 1)
    canvas = np.zeros((1200, 1800, 3), np.uint8)
    canvas.fill(255)
    model = HandGestureDetector(cap, hands, mpHands, detector, canvas)
    model.run_detection()

