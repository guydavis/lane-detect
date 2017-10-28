#
# Attempting to replicate lane detection results described in this tutorial by Naoki Shibuya:
# https://medium.com/towards-data-science/finding-lane-lines-on-the-road-30cf016a1165
# For more see: https://github.com/naokishibuya/car-finding-lane-lines
# 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
import subprocess
import os
import shutil
import traceback
from moviepy.editor import *
from collections import deque

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
    return cv2.bitwise_and(image, mask)
    
def select_region(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.90]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.90]
    top_right    = [cols*0.6, rows*0.6] 
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def hough_lines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

def average_slope_intercept(lines):
    left_lines    = [] 
    left_weights  = [] 
    right_lines   = [] 
    right_weights = [] 
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue 
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: 
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    return left_lane, right_lane 

def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0] 
    y2 = y1*0.6         
    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    return left_line, right_line
    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def mark_failed(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "DETECT FAILED!"
    textsize = cv2.getTextSize(text, font, 2, 5)[0]
    textX = int((image.shape[1] - textsize[0]) / 2)
    textY = int((image.shape[0] + textsize[1]) / 2)
    cv2.putText(image, text, (textX, textY), font, 2, (255, 0, 0), 5)
    return image

def process_image(dirpath, image_file):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    if not os.path.exists('output'):
        os.makedirs('output')
    image_name = os.path.splitext(image_file)[0]

    # First load and show the sample image
    image = mpimg.imread("{0}/{1}".format(dirpath, image_file))
    im = plt.imshow(image)
    plt.savefig('tmp/1.png')

    # Now select the white and yellow lines
    white_yellow = select_white_yellow(image)
    im = plt.imshow(white_yellow, cmap='gray')
    plt.savefig('tmp/2.png')

    # Now convert to grayscale
    gray_scale = convert_gray_scale(white_yellow)
    im = plt.imshow(gray_scale, cmap='gray')
    plt.savefig('tmp/3.png')

    # Then apply a Gaussian blur
    blurred_image = apply_smoothing(gray_scale)
    im = plt.imshow(blurred_image, cmap='gray')
    plt.savefig('tmp/4.png')

    # Detect line edges 
    edged_image = detect_edges(blurred_image)
    im = plt.imshow(edged_image, cmap='gray')
    plt.savefig('tmp/5.png')

    # Now ignore all but the area of interest
    masked_image = select_region(edged_image)
    im = plt.imshow(masked_image, cmap='gray')
    plt.savefig('tmp/6.png')
    
     # Apply Houghed lines algorithm
    houghed_lines = hough_lines(masked_image)
    if houghed_lines is not None:
        houghed_image = draw_lane_lines(image, lane_lines(image, houghed_lines))
        im = plt.imshow(houghed_image, cmap='gray')
        output_name = "output/{0}_passed.gif".format(image_name)
        print("Detected lanes in '{0}/{1}'. See result in '{2}'.".format(dirpath, image_file, output_name))
    else:
        im = plt.imshow(mark_failed(image), cmap='gray')
        output_name = "output/{0}_failed.gif".format(image_name)
        print("Failed detection in '{0}/{1}'. See result in '{2}'.".format(dirpath, image_file, output_name))
    plt.savefig('tmp/7.png')

    # Repeat last image in the loop a couple of times.
    plt.savefig('tmp/8.png')
    plt.savefig('tmp/9.png')

    # Now generate an animated gif of the image stages
    subprocess.call( ['convert', '-delay', '100', '-loop', '0', 'tmp/*.png', output_name] )
    shutil.rmtree('tmp')

QUEUE_LENGTH=50
class LaneDetector:
    def __init__(self):
        self.left_lines  = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def mean_line(self, line, lines):
        if line is not None:
            lines.append(line)
        if len(lines)>0:
            line = np.mean(lines, axis=0, dtype=np.int32)
            line = tuple(map(tuple, line))
        return line

    def process(self, image):
        try:
            white_yellow = select_white_yellow(image)
            gray         = convert_gray_scale(white_yellow)
            smooth_gray  = apply_smoothing(gray)
            edges        = detect_edges(smooth_gray)
            regions      = select_region(edges)
            lines        = hough_lines(regions)
            left_line, right_line = lane_lines(image, lines)
            left_line  = self.mean_line(left_line,  self.left_lines)
            right_line = self.mean_line(right_line, self.right_lines)
            return draw_lane_lines(image, (left_line, right_line))
        except:
            #traceback.print_exc()
            return image

def process_video(dirpath, video_file):
    video_outfile = os.path.splitext(video_file)[0] + '.mp4'
    detector = LaneDetector()
    clip = VideoFileClip(os.path.join(dirpath, video_file))
    processed = clip.fl_image(detector.process)
    print(os.path.join('output', video_file))
    processed.write_videofile(os.path.join('output', video_outfile), audio=False)

if __name__ == "__main__": 
    if len(sys.argv) == 1:
        print("Usage: python3 ./lane_detect.py images/* videos/*")
    else:
        for arg in sys.argv[1:]:
            if not os.path.isfile(arg):
                print("Not a file: {0}".format(arg))
            else:
                dirpath,filename = os.path.split(arg)
                if (dirpath.endswith('images')):
                    process_image(dirpath, filename)
                elif (dirpath.endswith('videos')):
                    process_video(dirpath, filename)
                else:
                    print("ERROR: Provide filenames in either images or videos folders.")
