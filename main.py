"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


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

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# People detection statistics metrics
total_people_count = 0
current_people_count = 0
current_entry_time = 0
current_exit_time = 0
current_duration = 0
potential_exit = False

def people_counter_update(count, frame_time, time_gap_threshold):
    """
    Update statistics per frame
    To counter mis-detection, a predetermined time windows is used to test whther the
    disappearance of a people from frame is a real one. If a detection happens within
    the window that means the previous disappearance is a false one. Otherwise the
    previous exit is counted as a real one.
  
    input: 
    count: count of people in current frame
    frame_time: time stamp
    time_gap_threshold: predetermined threshold to deermine with an exit is real

    output:
    None
    """
    global total_people_count
    global current_people_count
    global current_entry_time
    global current_exit_time
    global current_duration
    global potential_exit
  
    # Detected people entering frame
    if count > current_people_count:
        if potential_exit:
            # Check if the gap is over threshold, if yes then real exit
            if (time.time() - current_exit_time) > time_gap_threshold:
                print("Exceed time gap. Real entry")
                potential_exit = False
                total_people_count = total_people_count + count - current_people_count
                current_entry_time = frame_time
            # else continue
            else:
                print("Under time gap. Continue")
                pass
        else:
            total_people_count = total_people_count + count - current_people_count
            current_entry_time = frame_time
        print("People enter. Total count: " + str(total_people_count) + " at time:" + str(current_entry_time))
    # Detected people exiting frame
    elif count < current_people_count:
        if not potential_exit:
            potential_exit = True
            current_exit_time = frame_time
            current_duration = current_exit_time - current_entry_time
            print("People exit. Duration: " + str(current_duration) + " at time:" + str(current_exit_time))        
    # No change in the number of people in frame
    else:
        if potential_exit and (time.time() - current_exit_time) > time_gap_threshold:
            potential_exit = False
            current_duration = current_exit_time - current_entry_time

    # Update counter for people currently in frame
    current_people_count = count

def draw_boxes(frame, result, prob_threshold, width, height):
    """
    Draw bounding boxes to the frame
    
    input:
    frame: frame from camera/video
    result: list contains the result of inference
    
    output:
    frame: frame with bounding box drawn on it
    """
    start_point = None
    end_point = None
    thickness = 5
    color = (0, 255, 0)
    
    # Draw bounding box around detected person
    for box in result[0][0]: 
        if box[2] > prob_threshold:
            start_point = (int(box[3] * width), int(box[4] * height))
            end_point = (int(box[5] * width), int(box[6] * height))
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
            box_label = '{}: {:.2f}%'.format("people", box[2] * 100)
            frame = cv2.putText(frame, box_label , (int(box[3] * width)+ 5, int(box[4] * height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 86, 0), 2)
            
    return frame

def person_count_per_frame(result, prob_threshold):
    """
    Counts number of people in a frame based on confidence threshold

    input:
    result: inference results
    prob_threshold: detenction probablity thresholds

    output: Total number of people in frame
    """ 
    return sum (1 for i in result if i[0][0][2] > prob_threshold)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    model = args.model
    device = args.device
    infer_network.load_model(model, device)
    network_shape = infer_network.get_input_shape()

    #print(network_shape)

    input_shape = network_shape['image_tensor']

    ### TODO: Handle the input stream ###
    single_image_mode = False

    # if the input is a live feed such as webcam, set import to port 0
    if args.input == 'CAM':
        inference_input = 0
    else:
        inference_input = args.input
        assert os.path.isfile(inference_input), "[ERROR]: Video or image file cannot be found!"

        # Check to see if the input is single image or cideo file
        if args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png') or args.input.endswith('.gif'):
            single_image_mode = True
        elif args.input.endswith('.mp4') or args.input.endswith('.avi') or args.input.endswith('.mpg'):
            single_image_mode = False
            #print("Video file")
        #else:
        #    asssert 1, "[ERROR]: Unknown input file format!"        

    # Specify the timeing gap to filter out false mis-detections
    if prob_threshold <= 0.4:
        time_gap_threshold = 4
    elif prob_threshold <= 0.6:
        time_gap_threshold = 10
    else:
        time_gap_threshold = 15
    print("Time gap threshold: " + str(time_gap_threshold))  
    
    cap = cv2.VideoCapture(inference_input)
    cap.open(inference_input)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    if not single_image_mode:
        out_video = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 30, (width,height))

    ### TODO: Loop until stream is over ###
    print("Starting inferencing..")
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        #cv2.imshow('Frame',frame)
 
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        network_input = {'image_tensor': p_frame, 'image_info': p_frame.shape[1:]}
        report_duration = None
        infer_start = time.time()
        infer_network.exec_net(request_id = 0, network_input = network_input)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            person_count = 0

            ### TODO: Get the results of the inference request ###
            infer_time_diff = time.time() - infer_start
            output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            bounded_frame = draw_boxes(frame, output, prob_threshold, width, height)
            infer_time_text = "Inference time: {:.3f}ms".format(infer_time_diff * 1000)
            bounded_frame = cv2.putText(bounded_frame, infer_time_text, (15,15), cv2.FONT_HERSHEY_COMPLEX,0.45, (255, 86, 0), 1)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            current_people_count = person_count_per_frame(output,args.prob_threshold)
            current_time = time.time()

            # run update function per frame
            people_counter_update(current_people_count, current_time, time_gap_threshold)

            client.publish("person/duration", json.dumps({"duration" : 10}))
            client.publish("person", json.dumps({"count" : 8, "total" : 8}))           
            #client.publish("person", json.dumps({"count" : report_count, "total" : total_count}), qos = 0, retain = False)
            #client.publish("person", json.dumps({"count" : report_count}), qos = 0, retain = False) 
            #if report_duration is not None:
            #    client.publish("person/duration", json.dumps({"duration" : report_duration}), qos = 0, retain = False)

        ### TODO: Send the frame to the FFMPEG server ###
            bounded_frame = cv2.resize(bounded_frame, (width, height))
            #sys.stdout.buffer.write(bounded_frame)
            #sys.stdout.flush()
            cv2.imshow('Frame', bounded_frame)

        ### TODO: Write an output image if `single_image_mode` ###
            if single_image_mode:
                path = '.'
                cv2.imwrite(os.path.join(path , output_file ), bounded_frame)
            else:
                #print("Writing to video file")
                out_video.write(bounded_frame)

    print("Complete at: " + str(time.time()))

    # All done. Clean up
    out_video.release()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

 
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
