import time
import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np

class HandGestureDetector:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.offset = 20
        self.imgsize = 300
        self.count = 0
        self.canvas= np.zeros((720, 2000, 3), np.uint8)
        self.canvas.fill(255)
        self.x= 0
        self.y= 0
        self.prev_x= 0
        self.prev_y= 0
        self.drawing= False

    def find_hands(self, img):
        hand, img = self.detector.findHands(img)
        img = cv2.flip(img, 1)
        return hand, img


    def process_hands(self, hand, results, img):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP]
                cx, cy = int(index_finger_tip.x * img.shape[1]), int(index_finger_tip.y * img.shape[0])
                self.prev_x = cx
                self.prev_y = cy

                if hand:
                    handType = hand[0]['type']
                    if handType == "Left":
                        self.drawing = False
                        cv2.circle(self.canvas, (cx, cy), 150, (255, 255, 255), cv2.FILLED)
                    elif handType == 'Right':

                        if self.drawing:
                            cv2.line(self.canvas, (self.x, self.y), (cx, cy), (0), thickness=10)
                        # cv2.circle(self.canvas, (self.prev_x, self.prev_y), 10,(0),cv2.FILLED )
                        self.x = cx
                        self.y = cy


                        self.drawing = True

    def show_image(self, img):
        img = np.hstack((img, self.canvas))
        cv2.imshow("image", img)
        cv2.waitKey(1)

    def run_detection(self):
        while True:
            success, img = self.cap.read()
            hand, img = self.find_hands(img)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            self.process_hands(hand, results, img)
            self.show_image(img)
model = HandGestureDetector()
model.run_detection()
