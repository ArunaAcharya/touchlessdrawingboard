import time
import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import joblib


class DrawingApp:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.offset = 20
        self.labels = ['draw', 'erase']
        self.folder = "Data/erase"
        self.count = 0
        self.imgsize = 300

    def run(self):
        while True:
            success, img = self.cap.read()
            hands, img = self.detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgcrop = img[y-self.offset:y + h +self.offset, x- self.offset : x + w + self.offset]

                white = np.ones((self.imgsize, self.imgsize, 3), np.uint8)
                white.fill(255)
                imageShape = imgcrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = self.imgsize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgcrop, (wCal, self.imgsize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((self.imgsize - wCal) / 2)
                    white[:, wGap:wCal+wGap] = imgResize
                else:
                    k = self.imgsize / w
                    wCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgcrop, (self.imgsize, wCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((self.imgsize - wCal) / 2)
                    white[hGap: wCal + hGap, :] = imgResize

                cv2.imshow("white", white)
                cv2.imshow("crop image", imgcrop)

            cv2.imshow("image", img)
            cv2.waitKey(1)

            key = cv2.waitKey(1)
            if key == ord("s"):
                self.count += 1
                cv2.imwrite(f"{self.folder}/Image_{time.time()}.jpg", white)
                print(self.count)

            if key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DrawingApp()
    app.run()
