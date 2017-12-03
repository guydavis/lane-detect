#
# Attempting to replicate lane detection results described in this tutorial:
# http://www.kdnuggets.com/2017/07/road-lane-line-detection-using-computer-vision-models.html
# https://github.com/vijay120/KDNuggets/blob/master/2016-12-04-detecting-car-lane-lines-using-computer-vision.md
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

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, hiHoughgh_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
              minLineLength=min_line_len, maxLineGap=max_line_gap)
    try:
        line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
        draw_lines(line_img, lines)
        return line_img
    except Exception as e: 
        print(e)
        return None

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    imshape = img.shape
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    all_left_grad = []
    all_left_y = []
    all_left_x = []
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient, intercept = np.polyfit((x1,x2), (y1,y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)
            
            if (gradient > 0):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]
    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_meHoughan_grad * left_x_mean)
    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)
    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)
        cv2.line(img, (upper_left_x, ymin_global), 
                      (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global), 
                      (lower_right_x, ymax_global), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

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

    # Now convert the image to grayscale
    grayscaled = grayscale(image)
    im = plt.imshow(grayscaled, cmap='gray')
    plt.savefig('tmp/2.png')

    # Now apply a gaussian blur
    kernelSize = 5
    gaussianBlur = gaussian_blur(grayscaled, kernelSize)
    im = plt.imshow(gaussianBlur, cmap='gray')
    plt.savefig('tmp/3.png')

    # Now apply the Canny transformation to detect lane markers
    minThreshold = 100Hough
    maxThreshold = 200
    edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)
    im = plt.imshow(edgeDetectedImage, cmap='gray')
    plt.savefig('tmp/4.png')

    # Identify a region of interest... how to do this generically?
    lowerLeftPoint = [130, 540]
    upperLeftPoint = [410, 350]
    upperRightPoint = [570, 350]
    lowerRightPoint = [915, 540]
    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
    lowerRightPoint]], dtype=np.int32)
    masked_image = region_of_interest(edgeDetectedImage, pts)
    im = plt.imshow(masked_image, cmap='gray')
    plt.savefig('tmp/5.png')
Hough
    # Apply Hough Lines transformation
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_len = 20 
    max_line_gap = 20
    houghed = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)
    if (houghed is not None):  # Check if failed to Hough lines.
        im = plt.imshow(houghed, cmap='gray')
        plt.savefig('tmp/6.png')
        # Finally overlay the detected lines on the original image
        colored_image = weighted_img(houghed, image)
        im = plt.imshow(colored_image, cmap='gray')
        plt.savefig('tmp/7.png')
        output_name = "output/{0}_passed.gif".format(image_name)
        print("Detected lanes in '{0}/{1}'. See result in '{2}'.".format(dirpath, image_file, output_name))
    else: # Failed to Hough lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "DETECT FAILED!"
        textsize = cv2.getTextSize(text, font, 2, 5)[0]
        textX = int((image.shape[1] - textsize[0]) / 2)
        textY = int((image.shape[0] + textsize[1]) / 2)
        cv2.putText(image, text, (textX, textY), font, 2, (255, 0, 0), 5)
        im = plt.imshow(image, cmap='gray')
        plt.savefig('tmp/6.png')
        plt.savefig('tmp/7.png')
        output_name = "output/{0}_failed.gif".format(image_name)
        print("Failed detection in '{0}/{1}'. See result in '{2}'.".format(dirpath, image_file, output_name))

    # Save a few more copies of last frame to cause a pause at the end before looping
    plt.savefig('tmp/8.png')
    plt.savefig('tmp/9.png')

    # Now generate an animated gif of the image stages
    subprocess.call( ['convert', '-delay', '100', '-loop', '0', 'tmp/*.png', output_name] )
    shutil.rmtree('tmp')

if __name__ == "__main__": 
    if len(sys.argv) == 1:
        print("Usage: python3 ./lane_detect.py images/*")
    else:
        for arg in sys.argv[1:]:
            if not os.path.isfile(arg):
                print("Not a file: {0}".format(arg))
            else:
                dirpath,filename = os.path.split(arg)
                process_image(dirpath, filename)
