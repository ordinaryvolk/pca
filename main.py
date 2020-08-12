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
total_duration = 0
potential_exit = False
frame_count = 0
confirmed_exit = False

# Time per frame in seconds based on ffmeg return of video file: ffmpeg -i <File>
time_per_frame = 0.1

def people_counter_update(count, frame_time, time_gap_threshold):
    """
    Update statistics per frame
    
    To counter mis-detection, a predetermined time windows is used to test whther the
    disappearance of a people from frame is a real one. If a detection happens within
    the window that means the previous disappearance is a false one. Otherwise the
    pending exit is counted as a real one and duration will then be published to MTTQ. 
    The width of the window is determined by the detection probablity threshold. 
  
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
    global total_duration
    global potential_exit
    global confirmed_exit

    # Detected people entering frame
    if count > current_people_count:
        if potential_exit:
            # Check if the gap is over threshold, if yes then real exit
            if (frame_time - current_exit_time) > time_gap_threshold:
                #print("Exceed time gap. Real entry")
                potential_exit = False
                total_people_count = total_people_count + count - current_people_count
                #total_duration = total_duration + (current_exit_time - current_entry_time)* time_per_frame 
                total_duration = (current_exit_time - current_entry_time)* time_per_frame
                confirmed_exit = True
                current_entry_time = frame_time
                current_exit_time = None
            # else continue
            else:
                #print("Under time gap. Continue")
                current_exit_time = None
                potential_exit = False
        else:
            total_people_count = total_people_count + count - current_people_count
            current_entry_time = frame_time
            current_exit_time = None
            #print("People enter. Total count: " + str(total_people_count) + " at time:" + str(current_entry_time))
    # Detected people exiting frame
    elif count < current_people_count:
        potential_exit = True
        current_exit_time = frame_time
        #print("Pending exit.  "  + " at time:" + str(current_exit_time))        
    # No change in the number of people in frame
    else:
        if potential_exit and (frame_time - current_exit_time) > time_gap_threshold:
            potential_exit = False
            #total_duration = total_duration + (current_exit_time - current_entry_time)* time_per_frame
            total_duration = (current_exit_time - current_entry_time)* time_per_frame
            confirmed_exit = True
            current_exit_time = None

    # Update counter for people currently in frame
    current_people_count = count

def draw_bounding_boxes(frame, result, prob_threshold, width, height):
    """
    Draw bounding boxes to the frame
    
    input:
    frame: input frame 
    result: inferencing results
    
    output:
    frame: frame with bounding boxes
    """
    thickness = 2
    color = (0, 255, 0)
    
    # Draw bounding box around detected person
    for detected in result[0][0]: 
        if detected[2] > prob_threshold:
            xmin = int(detected[3] * width)
            ymin = int(detected[4] * height)
            xmax = int(detected[5] * width)
            ymax = int(detected[6] * height)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
            label = '{}: {:.2f}%'.format("People", detected[2] * 100)
            frame = cv2.putText(frame, label , (xmin , ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 86, 0), 2)
            
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
    global frame_count
    global confirmed_exit

    # use fixed request id
    current_request_id = 0

    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    model = args.model
    device = args.device
    cpu_extension = args.cpu_extension
    infer_network.load_model(model, device, cpu_extension, current_request_id)
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
        time_gap_threshold = 10 
    elif prob_threshold <= 0.6:
        time_gap_threshold = 30 
    else:
        time_gap_threshold = 40
    #print("Time gap threshold: " + str(time_gap_threshold))  
    
    cap = cv2.VideoCapture(inference_input)
    cap.open(inference_input)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    if not single_image_mode:
        out_video = cv2.VideoWriter('./output/out.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 30, (width,height))

    ### TODO: Loop until stream is over ###
    #print("Starting inferencing..")
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        #cv2.imshow('Frame',frame)
 
        ### TODO: Pre-process the image as needed ###
        preprossed_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        preprossed_frame = preprossed_frame.transpose((2,0,1))
        b = 1
        c = preprossed_frame.shape[0]
        h = preprossed_frame.shape[1]
        w = preprossed_frame.shape[2]
        preprossed_frame = preprossed_frame.reshape(b, c, h, w)
        
        ### TODO: Start asynchronous inference for specified request ###
        network_input = {'image_tensor': preprossed_frame, 'image_info': preprossed_frame.shape[1:]}
        
        infer_start = time.time()
        infer_network.exec_net(request_id = current_request_id, network_input = network_input)

        frame_count = frame_count + 1
        last_total_people_count = total_people_count

        ### TODO: Wait for the result ###
        if infer_network.wait(current_request_id) == 0:
            person_count = 0
            
            ### TODO: Get the results of the inference request ###
            infer_time_diff = time.time() - infer_start
            output = infer_network.get_output(current_request_id)

            ### TODO: Extract any desired stats from the results ###
            bounded_frame = draw_bounding_boxes(frame, output, prob_threshold, width, height)
            infer_time_text = "Inference time: {:.3f}ms".format(infer_time_diff * 1000)
            bounded_frame = cv2.putText(bounded_frame, infer_time_text, (15,15), cv2.FONT_HERSHEY_COMPLEX,0.45, (255, 86, 0), 1)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            current_people_count = person_count_per_frame(output,args.prob_threshold)
            # run update function per frame
            people_counter_update(current_people_count, frame_count, time_gap_threshold)
            #print("People count: " + str(current_people_count) + " Total count: "+ str(total_people_count) + " Last total count: " + str(last_total_people_count) + " Duration: " + str(total_duration) + " at frame: " + str(frame_count))

            # Publish current people count every frame        
            client.publish("person", json.dumps({"count" : current_people_count}))
            # Because of the UI problem, i.e. total count will increase when duration is
            # published, the duration is only published after a confirmed exit. That means
            # the statistics in UI will be updated at a time after the actual exit happens
            if confirmed_exit:
                #if last_total_people_count==0:
                #    client.publish("person/duration", json.dumps({"duration" : 0}))                           # else:
                client.publish("person/duration", json.dumps({"duration" : total_duration})) # Work around UI counting problem
                confirmed_exit = False
          
                    #print("Publiching average duration: " + str(total_duration/last_total_people_count) + " based on total: " + str(last_total_people_count))
        ### TODO: Send the frame to the FFMPEG server ###
            bounded_frame = cv2.resize(bounded_frame, (width, height))
            sys.stdout.buffer.write(bounded_frame)
            sys.stdout.flush()
            #cv2.imshow('Frame', bounded_frame)

        ### TODO: Write an output image if `single_image_mode` ###
            if single_image_mode:
                path = './output'
                cv2.imwrite(os.path.join(path , output_file ), bounded_frame)
            else:
                #print("Writing to video file")
                out_video.write(bounded_frame)

    #print("Complete at: " + str(time.time()))

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
