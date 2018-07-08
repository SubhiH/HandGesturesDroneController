# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.cluster.vq import vq, kmeans
import time as t


detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27

# MODEL_NAME = 'hand_inference_graph'
# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')
#
# NUM_CLASSES = 1
#




# What model to download.
MODEL_NAME = 'hand_detector_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 6


# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes ,classes , im_width, im_height, image_np,prev_p1,prev_p2):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))


            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 2, 1)
            cv2.putText(image_np,str(classes[i]),(int(left)-5, int(top)-5),cv2.FONT_HERSHEY_SIMPLEX,2,255)

            width = right - left;
            height = bottom - top;

            prev_width = prev_p2[0] - prev_p1[0];
            prev_height = prev_p2[1] - prev_p1[1];


            cv2.circle(image_np,(int(left+width/4),int(top+height/4)), 2, (0,0,255), -1)
            cv2.circle(image_np,(int(right-width/4),int(top+height/4)), 2, (0,0,255), -1)
            cv2.circle(image_np,(int(left+width/4),int(bottom-height/4)), 2, (0,0,255), -1)
            cv2.circle(image_np,(int(right-width/4),int(bottom-height/4)), 2, (0,0,255), -1)
            cv2.circle(image_np,(int(right-width/2),int(bottom-height/2)), 2, (0,0,255), -1)

            if not prev_p1[0]==0 and not prev_p1[1]==0:
                cv2.line(image_np,(int(prev_p1[0]+prev_width/4),int(prev_p1[1]+prev_height/4)),(int(left+width/4),int(top+height/4)),(255,0,0),1)


            box_area = image_np[int(top):int(bottom),int(left):int(right),:];
            hsv_box_area = cv2.cvtColor(box_area,cv2.COLOR_BGR2HSV);

            ##plt.ion()

            # mean = np.mean(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,:],axis=2)#=255
            mean_h = np.mean(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,0])#=255
            mean_s = np.mean(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,1])
            mean_v = np.mean(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,2])
            print 'mean : ',mean_h,mean_s,mean_v


            min_h = np.min(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,0])#=255
            min_s = np.min(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,1])
            min_v = np.min(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,2])
            # print 'min : ',min_h,min_s,min_v

            max_h = np.max(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,0])#=255
            max_s = np.max(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,1])
            max_v = np.max(box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,2])
            # print 'max : ',max_h,max_s,max_v

            hue = box_area[(box_area.shape[0]/2):(box_area.shape[0]/2)+80,(box_area.shape[1]/2)-20:(box_area.shape[1]/2)+20,0]


            cv2.imshow("box",cv2.cvtColor(box_area, cv2.COLOR_RGB2BGR));
            ##plt.clf()



            # color = ('b','g','r')
            # for i,col in enumerate(color):
            #     histr = cv2.calcHist([hsv_box_area],[i],None,[256],[0,256])
            #     np.max(histr)
            #     plt.plot(histr,color = col)
            #     plt.xlim([0,256])
            #     plt.show()
            hue, sat, val = hsv_box_area[:,:,0], hsv_box_area[:,:,1], hsv_box_area[:,:,2]


            ##plt.close("all")

            img,h,s,v = do_cluster(hsv_box_area, 3, 3)

            # org_img = cv2.cvtColor(box_area, cv2.COLOR_HSV2BGR);

            cv2.imshow("clusters",cv2.cvtColor(img, cv2.COLOR_HSV2RGB));
            # cv2.imshow("img_s",cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB));

            # Normal masking algorithm
            lower_skin = np.array([mean_h-45,s-50,0])
            upper_skin = np.array([mean_h+45,s+50,255])
            # print h,s

            mask = cv2.inRange(hsv_box_area,lower_skin, upper_skin)
            result = cv2.bitwise_and(box_area,box_area,mask = mask)

            cv2.imshow('result',result)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            im_bw = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)[1]

            cv2.imshow('result1',im_bw)




                # new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
                # plt.imshow(new_image)
                
            # plt.show()
            

            prev_p1 = p1;
            prev_p2 = p2;
            return p1,p2
        return (0,0),(0,0)


def calculate_intersection(box_1,box_2,im_width,im_height):
    (left_1, right_1, top_1, bottom_1) = (box_1[1] * im_width, box_1[3] * im_width,
                                              box_1[0] * im_height, box_1[2] * im_height)

    (left_2, right_2, top_2, bottom_2) = (box_2[1] * im_width, box_2[3] * im_width,
                                              box_2[0] * im_height, box_2[2] * im_height)

    area_2 = (right_2-left_2)*(bottom_2-top_2)
    x = max(left_1, left_2)
    y = max(top_1, top_2)
    if x <= min(right_1, right_2):
        w = min(right_1, right_2) - x
        h = min(bottom_1, bottom_2) - y
    else:
        w=0
        h=0
    return x,y,w,h,

def is_hand_opened(hand_class):
    if hand_class == 5.0 or hand_class == 4.0 or hand_class == 3.0:
        return 1
    else:
        return 0

def draw_steering_wheel(img,rotation):
    raduis = min(img.shape[0],img.shape[1]);
    raduis = (raduis - raduis/6)/2;
    wheel_color = (200,200,200);
    shift_from_center = 55
    # rotation = np.abs(rotation)
    # if rotation >= shift_from_center:
    #     dest = 0
    # else:
    #     dest = np.sqrt(shift_from_center*shift_from_center-rotation*rotation)
    # print shift_from_center,rotation,dest
    overlay = img.copy()
    # (2) draw shapes:
    # (3) blend with the original:
    opacity = 0.4
    
    cv2.circle(overlay,(int(img.shape[1]/2),int(img.shape[0]/2)), raduis,wheel_color , 3)
    cv2.circle(overlay,(int(img.shape[1]/2),int(img.shape[0]/2)), raduis-25,wheel_color, 3)
    cv2.line(overlay,(int((img.shape[1]/2)-shift_from_center),int(img.shape[0]/2)+rotation),(int(((img.shape[1]/2))+shift_from_center),int(img.shape[0]/2)-rotation),wheel_color , 3)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    

    # pts = np.array([[int(((img.shape[1]/2))+dest),int(img.shape[0]/2)+rotation]
    # ,[int(((img.shape[1]/2))+dest-10),int(img.shape[0]/2)-10]+rotation,
    # [int(((img.shape[1]/2))+dest-10),int(img.shape[0]/2)+10]]+rotation,
    # np.int32)
    # pts = pts.reshape((-1,1,2))
    # cv2.polylines(img,[pts],True,wheel_color,3,-1)
    # if rotation>=0:
        # cv2.line(img,(int((img.shape[1]/2)-raduis+25+(rotation/4.2)),int(img.shape[0]/2)-10-rotation),(int((img.shape[1]/2)+raduis-25-rotation/3.8),int(img.shape[0]/2)-10+rotation),wheel_color , 3)
        # cv2.line(img,(int((img.shape[1]/2)-raduis+25+(rotation/3.8)),int(img.shape[0]/2)+10-rotation),(int((img.shape[1]/2)+raduis-25-rotation/4.2),int(img.shape[0]/2)+10+rotation),wheel_color , 3)
        # cv2.line(img,(int((img.shape[1]/2)-25),int(img.shape[0]/2)),(int(((img.shape[1]/2)+25)),int(img.shape[0]/2)+10+rotation),wheel_color , 3)
    # else:
        # cv2.line(img,(int((img.shape[1]/2)-raduis+25-(rotation/4.2)),int(img.shape[0]/2)-10-rotation),(int((img.shape[1]/2)+raduis-25+rotation/3.8),int(img.shape[0]/2)-10+rotation),wheel_color , 3)
        # cv2.line(img,(int((img.shape[1]/2)-raduis+25-(rotation/3.8)),int(img.shape[0]/2)+10-rotation),(int((img.shape[1]/2)+raduis-25+rotation/4.2),int(img.shape[0]/2)+10+rotation),wheel_color , 3)

    return img


def draw_right_arrow(img,shift_arrow):
    wheel_color = (200,200,200)
    shift_from_center = 55
    overlay = img.copy()
    # (2) draw shapes:
    # (3) blend with the original:
    opacity = 0.7
    
    cv2.line(overlay,(int((img.shape[1]/2)-shift_from_center+shift_arrow*5),int(img.shape[0]/2)),(int(((img.shape[1]/2))+shift_from_center+shift_arrow*5),int(img.shape[0]/2)),wheel_color , 15)
    pts = np.array([[int(((img.shape[1]/2))+shift_from_center+shift_arrow*5)+25,int(img.shape[0]/2)]
    ,[int(((img.shape[1]/2))+shift_from_center+shift_arrow*5),int(img.shape[0]/2)-25],
    [int(((img.shape[1]/2))+shift_from_center+shift_arrow*5),int(img.shape[0]/2)+25]],
    np.int32)
    pts = pts.reshape((-1,1,2))
    # cv2.fillPoly(img,[pts],wheel_color,-1)
    cv2.fillPoly(overlay, [pts], wheel_color, 8)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    return img

def draw_left_arrow(img,shift_arrow):
    wheel_color = (200,200,200)
    shift_from_center = 55
    overlay = img.copy()
    # (2) draw shapes:
    # (3) blend with the original:
    opacity = 0.7
    
    cv2.line(overlay,(int((img.shape[1]/2)-shift_from_center-shift_arrow*5),int(img.shape[0]/2)),(int(((img.shape[1]/2))+shift_from_center-shift_arrow*5),int(img.shape[0]/2)),wheel_color , 15)
    pts = np.array([[int(((img.shape[1]/2))-shift_from_center-shift_arrow*5)-25,int(img.shape[0]/2)]
    ,[int(((img.shape[1]/2))-shift_from_center-shift_arrow*5),int(img.shape[0]/2)-25],
    [int(((img.shape[1]/2))-shift_from_center-shift_arrow*5),int(img.shape[0]/2)+25]],
    np.int32)
    pts = pts.reshape((-1,1,2))
    # cv2.fillPoly(img,[pts],wheel_color,-1)
    cv2.fillPoly(overlay, [pts], wheel_color, 8)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    return img

def check_pattern(list_1, list_2, pattern):
    array_list_1 = np.array(list_1)
    array_list_2 = np.array(list_2)
    array_pattern = np.array(pattern)
    if np.array_equal((array_list_1 | array_list_2),array_pattern):
        return True;
    else:
        return False;

def do_cluster(hsv_image, K, channels):
    # gets height, width and the number of channes from the image shape
    h,w,c = hsv_image.shape
    # prepares data for clustering by reshaping the image matrix into a (h*w) x c matrix of pixels
    cluster_data = hsv_image.reshape( (h*w,c) )
    # grabs the initial time
    t0 = t.time()
    # performs clustering
    codebook, distortion = kmeans(np.array(cluster_data[:,0:channels], dtype=np.float), K)
    # takes the final time
    t1 = t.time()
    print "Clusterization took %0.5f seconds" % (t1-t0)
    
    
    # calculates the total amount of pixels
    tot_pixels = h*w
    # generates clusters
    data, dist = vq(cluster_data[:,0:channels], codebook)
    # calculates the number of elements for each cluster
    weights = [len(data[data == i]) for i in range(0,K)]
    
    # creates a 4 column matrix in which the first element is the weight and the other three
    # represent the h, s and v values for each cluster
    color_rank = np.column_stack((weights, codebook))
    # sorts by cluster weight
    color_rank = color_rank[np.argsort(color_rank[:,0])]
    # print color_rank
    # print color_rank[::-1]

    # creates a new blank image
    new_image =  np.array([0,0,255], dtype=np.uint8) * np.ones( (200, 200, 3), dtype=np.uint8)
    img_height = new_image.shape[0]
    img_width  = new_image.shape[1]

    # for each cluster
    for i,c in enumerate(color_rank[::-1]):
        # gets the weight of the cluster
        weight = c[0]
        
        # calculates the height and width of the bins
        height = int(weight/float(tot_pixels) *img_height )
        width = img_width/len(color_rank)

        # calculates the position of the bin
        x_pos = i*width


        
        # defines a color so that if less than three channels have been used
        # for clustering, the color has average saturation and luminosity value
        color = np.array( [0,128,200], dtype=np.uint8)
        
        # substitutes the known HSV components in the default color
        for j in range(len(c[1:])):
            color[j] = c[j+1]
            # print color[j] , j
        
        # draws the bin to the image
        # print color[0], color[1]
        new_image[ img_height-height:img_height, x_pos:x_pos+width] = [color[0], color[1], color[2]]
        
    # returns the cluster representation
    # print color_rank[0]
    return new_image,int(color_rank[0][1]),int(color_rank[0][2]),int(color_rank[0][3])







# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores),np.squeeze(classes)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
