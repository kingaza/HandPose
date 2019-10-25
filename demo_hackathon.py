from utils import detector_utils as detector_utils
from utils import pose_classification_utils as classifier
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import os; 
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import gui

import numpy as np
import requests

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("hackathon.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)

logger.info('=' * 80)
logger.info("Start DEMO of Controlling Patient Table")
logger.info('=' * 80)

frame_processed = 0
score_thresh = 0.18

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue

url_root = 'http://md1z7xac.ad005.onehc.net:5757/api'


class ModeSwitch(object):

    def __init__(self):
        self.mode_light = False
        self.mode_ptab = False

        self.url_mode_ptab = url_root + '/mode/ptab'
        self.url_mode_light = url_root + '/mode/light'

    def set_ptab(self):
        self.mode_ptab = True
        self.mode_light = False
        logger.info('  ==> Set PTab Mode')
        resp = requests.get(self.url_mode_ptab)
        logger.info(f'Send request, receive: {resp.status_code}')        

    def set_light(self):
        self.mode_light = True
        self.mode_ptab = False
        logger.info('  ==> Set Light Mode')
        resp = requests.get(self.url_mode_light)
        logger.info(f'Send request, receive: {resp.status_code}')          



class LightController(object):
    def __init__(self):
        self.activated = False
        self.url_light_brighter = url_root + '/light/light-brighter'      
        self.url_light_darker = url_root + '/light/light-darker'           
        self.url_light_switch = url_root + '/light/light_switch'         
        self.url_light_pause = url_root + '/ptab/pause'   

        self.last_request_time = -1     

        self.in_darkering = False
        self.in_brightering = False

    def set_activated(self, activated):
        self.activated = activated

    def darker(self):
        if not self.in_darkering:
            logger.info('  ==> Darker Light')
            self.in_darkering = True
            resp = requests.get(self.url_light_darker)
            logger.info(f'Send request, receive: {resp.status_code}')
            self.last_request_time = time.time()   
        self.in_brightering = False    

    def brighter(self):
        if not self.in_brightering:
            self.in_brightering = True
            logger.info('  ==> Brighter Light')
            resp = requests.get(self.url_light_brighter)
            logger.info(f'Send request, receive: {resp.status_code}')
            self.last_request_time = time.time()  
        self.in_darkering = False       

    def pause(self):
        if self.in_brightering:
            logger.info('  ==> Pause brightering light')           
            self.in_brightering = False
            resp = requests.get(self.url_light_pause)
            logger.info(self.url_light_pause)
            logger.info(f'Send request, receive: {resp.status_code}') 
            self.last_request_time = time.time()  

        if self.in_darkering:
            logger.info('  ==> Pause darkering light')           
            self.in_darkering = False
            logger.info(self.url_light_pause)
            resp = requests.get(self.url_light_pause)
            logger.info(f'Send request, receive: {resp.status_code}')      
            self.last_request_time = time.time()             

    def switch(self):
        logger.info('  ==> Switch Light')
        resp = requests.get(self.url_light_switch)
        logger.info(f'Send request, receive: {resp.status_code}')
        self.last_request_time = time.time()                

class PTabController(object):
    def __init__(self):
        self.activated = False

        self.last_request_time = -1
        self.last_movein_time = -1
        self.last_moveout_time = -1

        self.in_tohome = False
        self.in_movein = False
        self.in_moveout = False

        self.url_ptab_tohome = url_root + '/ptab/move-home'      # thumb
        self.url_ptab_pause = url_root + '/ptab/pause'           # palm
        self.url_ptab_movein = url_root + '/ptab/move-in'        # left
        self.url_ptab_moveout = url_root + '/ptab/move-out'      # right    


    def set_activated(self, activated):
        self.activated = activated

    def move_in(self):
        if not self.in_movein:
            logger.info('  ==> Move PTab IN')
            resp = requests.get(self.url_ptab_movein)
            logger.info(f'Send request, receive: {resp.status_code}')
            self.last_request_time = time.time()   
        self.in_movein = True  
        self.in_moveout = False
        self.in_tohome = False     

    def move_out(self):
        if not self.in_moveout:
            logger.info('  ==> Move PTab OUT')
            resp = requests.get(self.url_ptab_moveout)
            logger.info(f'Send request, receive: {resp.status_code}') 
            self.last_request_time = time.time()   
        self.in_movein = False  
        self.in_moveout = True
        self.in_tohome = False  

    def to_home(self):
        if not self.in_tohome:
            logger.info('  ==> Move PTab to HOME')
            resp = requests.get(self.url_ptab_tohome)
            logger.info(f'Send request, receive: {resp.status_code}') 
            self.last_request_time = time.time()    
        self.in_movein = False  
        self.in_moveout = False
        self.in_tohome = True         

    def pause(self):
        if self.in_movein or self.in_moveout or self.in_tohome:
            logger.info('  ==> Pause PTab')
            resp = requests.get(self.url_ptab_pause)
            logger.info(f'Send request, receive: {resp.status_code}') 
            self.last_request_time = time.time()     
            self.in_movein = False  
            self.in_moveout = False
            self.in_tohome = False        



def worker(input_q, output_q, cropped_output_q, inferences_q, cap_params, frame_processed):
    logger.info(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)

    logger.info(">> loading keras model for worker")
    try:
        model, classification_graph, session = classifier.load_KerasGraph("cnn/models/handposes_vgg64_v1.h5")
    except Exception as e:
        logger.error(e)

    while True:
        #print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # get region of interest
            res = detector_utils.get_box_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
            
            # draw bounding boxes
            detector_utils.draw_box_on_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
            
            # classify hand pose
            if res is not None:
                class_res = classifier.classify(model, classification_graph, session, res)
                inferences_q.put(class_res)   
            
            # add frame annotated with bounding box to queue
            cropped_output_q.put(res)
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    input_q             = Queue(maxsize=args.queue_size)
    output_q            = Queue(maxsize=args.queue_size)
    cropped_output_q    = Queue(maxsize=args.queue_size)
    inferences_q        = Queue(maxsize=args.queue_size)

    video_capture = WebcamVideoStream(
        src=args.video_source, width=args.width, height=args.height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    logger.info(f"im_width={cap_params['im_width']}, im_height={cap_params['im_height']}")

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    logger.info(args)
    logger.info(cap_params)
    
    # Count number of files to increment new example directory
    poses = []
    _file = open("poses.txt", "r") 
    lines = _file.readlines()
    for line in lines:
        line = line.strip()
        if(line != ""):
            print(line)
            poses.append(line)

    logger.info(poses)        


    # spin up workers to paralleize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cropped_output_q, inferences_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    # cv2.namedWindow('Handpose', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Handpose', 0)
    cv2.resizeWindow('Handpose', 640, 360)


    switch = ModeSwitch()
    ptab = PTabController()
    light = LightController()

    switch_duration = 1.2
    waiting_duration = 0.2
    recognition_duration = 1.0

    # used for mode switching
    thumb_beginning_time = None
    fist_beginning_time = None

    no_inference_begining_time = None

    pose_buf = []
    time_buf = []

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        output_frame = output_q.get()
        cropped_output = cropped_output_q.get()

        inferences      = None

        try:
            inferences = inferences_q.get_nowait()      
        except Exception as e:
            pass      

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time


        if inferences is None:
            logger.debug('No hand detected')
            
            if no_inference_begining_time is None:
                no_inference_begining_time = time.time()
            
            if time.time() - no_inference_begining_time > waiting_duration:
                # None of Ptab and Light
                if switch.mode_ptab:
                    thumb_beginning_time = None

                if switch.mode_ptab:
                    fist_beginning_time = None

                # control PTab
                # Pause ptab moving if no request for a long time
                if switch.mode_ptab:
                    if time.time() - ptab.last_request_time > waiting_duration:
                        logger.debug('No request in the last waiting time')
                        logger.debug(f'PTab status: move-in={ptab.in_movein}, move-out={ptab.in_moveout}, tohome={ptab.in_tohome}')
                        if ptab.in_movein or ptab.in_moveout:
                            logger.info('Pause Ptab if it is not on the way home')
                            ptab.pause()

                if switch.mode_light:
                    if time.time() - light.last_request_time > waiting_duration:
                        logger.debug('No request in the last waiting time')
                        logger.debug(f'Light status: brightering={light.in_brightering}, darking={light.in_darkering}')
                        if light.in_brightering or light.in_darkering:
                            logger.info('Pause light brightering or darking')
                            light.pause()                            


        # Display inferences
        if(inferences is not None):
            logger.debug(inferences)

            no_inference_begining_time = None

            t = time.time()
            p = np.argmax(inferences)

            time_buf.insert(0, t)
            pose_buf.insert(0, p)

            # remove the data which is not in the time window of recognition
            for i in np.arange(len(time_buf)-1):
                if time_buf[0] - time_buf[-1] > recognition_duration:
                    time_buf.pop()
                    pose_buf.pop()

            if len(pose_buf) > 5:
                from collections import Counter
                c = Counter(pose_buf)
                most_common_pose, detect_times = c.most_common(1)[0]
                logger.info(f'Pose {poses[most_common_pose]} happens {detect_times} / {len(pose_buf)}')  

                # check firstly if switching mode needed
                if most_common_pose == 2:
                    if not fist_beginning_time:
                        fist_beginning_time = time.time()
                    thumb_beginning_time = None

                elif most_common_pose == 4:
                    if not thumb_beginning_time:
                        thumb_beginning_time = time.time()
                    fist_beginning_time = None

                else:    
                    thumb_beginning_time = None
                    fist_beginning_time = None


                # pose left
                if most_common_pose == 0:
                    if switch.mode_ptab:
                        ptab.move_in()
                    if switch.mode_light:
                        light.darker()    

                # pose right
                elif most_common_pose == 1:
                    if switch.mode_ptab:
                        ptab.move_out()
                    if switch.mode_light:
                        light.brighter()                        

                # pose fist, can switch to light mode
                elif most_common_pose == 2:
                    # mode switch?
                    if not switch.mode_light:
                        if time.time() - fist_beginning_time > switch_duration:
                            switch.set_light()

                # pose palm
                elif most_common_pose == 3:
                    # mode PAUSE: 
                    if switch.mode_ptab:
                        ptab.pause()

                # pose thumb, can switch to ptab mode
                elif most_common_pose == 4:
                    # mode switch?
                    if not switch.mode_ptab:
                        if time.time() - thumb_beginning_time > switch_duration:
                            switch.set_ptab()
                        
                    # MOVE HOME: 
                    if switch.mode_ptab:
                        ptab.to_home()

                else:
                    # Pause PTab except in the case of ToHome
                    if switch.mode_ptab:
                        if not ptab.in_tohome:
                            ptab.pause()

                    if switch.mode_light:
                        light.pause()                            


            gui.drawInferences(inferences, poses)

        if (cropped_output is not None):
            cropped_output = cv2.cvtColor(cropped_output, cv2.COLOR_RGB2BGR)
            if (args.display > 0):
                cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Cropped', 450, 300)
                cv2.imshow('Cropped', cropped_output)
                #cv2.imwrite('image_' + str(num_frames) + '.png', cropped_output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    logger.info(f'frames processed: {index} elapsed time: {elapsed_time}, fps: {str(int(fps))}')

    
        # print("frame ",  index, num_frames, elapsed_time, fps)

        if (output_frame is not None):
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            if (args.display > 0):
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)
                cv2.imshow('Handpose', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    logger.info(f'frames processed: {index} elapsed time: {elapsed_time}, fps: {str(int(fps))}')
        else:
            logger.info("video end")
            break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    logger.info(f'fps: {fps}')
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
