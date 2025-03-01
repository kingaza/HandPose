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
handler = logging.FileHandler("music.log")
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

    last_time = time.time()
    duration = 1.0
    pose_buf = []
    time_buf = []

    music_playing = False

    url_root = 'http://localhost:5000'
    url_music_play = url_root + '/music/play'
    url_music_stop = url_root + '/music/stop'
    url_music_last = url_root + '/music/last'
    url_music_next = url_root + '/music/next'

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

        # Display inferences
        if(inferences is not None):
            logger.debug(inferences)

            t = time.time()
            p = np.argmax(inferences)

            time_buf.insert(0, t)
            pose_buf.insert(0, p)

            # remove the data which is not in the time window
            for i in np.arange(len(time_buf)-1):
                if time_buf[0] - time_buf[-1] > duration:
                    time_buf.pop()
                    pose_buf.pop()

            if len(pose_buf) > 5:
                from collections import Counter
                c = Counter(pose_buf)
                most_common_pose, detect_times = c.most_common(1)[0]
                logger.info(f'Pose {poses[most_common_pose]} happens {detect_times} / {len(pose_buf)}')  
                
                # pose Palm
                if most_common_pose == 3:
                    if music_playing:
                        logger.info('  ==> Stop playing music')
                        resp = requests.get(url_music_stop)
                        logger.info(f'Send request, receive: {resp.status_code}') 
                        music_playing = False  

                # pose Thumb
                elif most_common_pose == 4:
                    if not music_playing:
                        logger.info('  ==> Play music')
                        resp = requests.get(url_music_play)
                        logger.info(f'Send request, receive: {resp.status_code}') 
                        music_playing = True  
                        last_time = t

                elif most_common_pose == 0:
                    logger.info('  ==> Play last music')
                    resp = requests.get(url_music_last)
                    logger.info(f'Send request, receive: {resp.status_code}') 
                    music_playing = True  
                    last_time = t            

                elif most_common_pose == 1:
                    logger.info('  ==> Play next music')
                    resp = requests.get(url_music_next)
                    logger.info(f'Send request, receive: {resp.status_code}') 
                    music_playing = True  
                    last_time = t                                      

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
