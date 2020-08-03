
import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network


def main():
    """
    Load the network and parse the output.

    :return: None
    """

    # Open video
    cap = cv2.VideoCapture('./resources/Pedestrian_Detect_2_1_1.mp4')
    cap.open('./resources/Pedestrian_Detect_2_1_1.mp4')

    # loop till it ends
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        print(frame.shape)
        #cv2.imshow('Frame',frame)
   
        # output every frame
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()


    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

