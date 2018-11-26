#coding=utf-8

import numpy as np
import glob
import cv2
from moviepy.editor import *
import matplotlib.pyplot as plt

chessBoardImageFiles = glob.glob('./camera_cal/calibration*.jpg')
projectVideo = './project_video.mp4'
output = './output_images/myOutput.mp4'

# Calibrate the camera first
# This cameraCalibration function will return the "ret, mtx, dist, rvecs, tvecs"
def cameraCalibraion(chessBoardImageFiles):
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    objpoints = [] 
    imgpoints = [] 
    
    # Step through the list and search for chessboard corners
    for fname in chessBoardImageFiles:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None) 
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # return ret, mtx, dist, rvecs, tvecs 
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# color transforms and gradients, will return a combined binary image.
def colorAndGradient(img, s_thresh=(150, 255), sx_thresh=(40, 100)):
    img = np.copy(img)

    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary

# warp the image into proper perspective and get a birds-eye-view
def warp(image, src, dst):
    imageSize = (image.shape[1], image.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, imageSize)

    return (warped, M, Minv)

# Define a mask apply to region of interest.
def regionOfInterest(image, vertices):
    mask = np.zeros_like(image)   
    
    if len(image.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
          
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# Find lane pixels, I use the histogram method. 
# Of course it's the simplest way, but I found its result is the best, 
# I also tried other two methods, can't get satisfied result
def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]//2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50
    window_height = np.int(binary_warped.shape[0]//nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low =  leftx_current - margin 
        win_xleft_high = leftx_current + margin  
        win_xright_low = rightx_current - margin  
        win_xright_high = rightx_current + margin  
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

# if the curverad of two lines(left and right) are different too much, 
# then will use the previous curverad.
g_left_curverad = 0
g_right_curverad = 0
g_left_fitx = 0
g_right_fitx = 0
def fit_polynomial(warped, Minv):
    global g_left_curverad, g_right_curverad, g_left_fitx, g_right_fitx

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)

    # according to my measurement, not very accurate.
    xm_per_pix = 3.7/600
    ym_per_pix = 30./780

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    real_left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    real_right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])

    y_eval = np.max(ploty)
    real_y_eval = ym_per_pix*y_eval
    
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    real_left_curverad = ((1 + (2*real_left_fit[0]*real_y_eval + real_left_fit[1])**2)**1.5) / np.absolute(2*real_left_fit[0])
    real_right_curverad = ((1 + (2*real_right_fit[0]*real_y_eval + real_right_fit[1])**2)**1.5) / np.absolute(2*real_right_fit[0])

    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    gap = max(right_fitx - left_fitx) - min(right_fitx - left_fitx)

    # 277 is not bad.
    if gap > 277:
        left_curverad = g_left_curverad
        right_curverad = g_right_curverad
        left_fitx = g_left_fitx
        right_fitx = g_right_fitx
    else:
        g_left_curverad = left_curverad
        g_right_curverad = right_curverad
        g_left_fitx = left_fitx
        g_right_fitx = right_fitx

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 

    #vehicle's offset of the middle of the lane.
    offset = (np.average(np.nonzero(newwarp[:,:,1]))-655) * xm_per_pix

    return newwarp,offset,real_left_curverad, real_right_curverad, gap, out_img


def drawLaneLines(image): 
    thisImage = np.copy(image)
    imshape = thisImage.shape

    # Undistort image according to the calibration parameters
    undistortImage = cv2.undistort(thisImage, mtx, dist, None, mtx)

    # Process image by color and gradient threshold
    colorGradThresImage = colorAndGradient(undistortImage)

    # Apply a mask to images, I used a complicated mask.
    vertices = np.array([[(100,imshape[0]),(550,450),(750,450),((imshape[1]-50),imshape[0]),(1060,720), (640,449),(320, 720)]], dtype=np.int32)
    maskedImage = regionOfInterest(colorGradThresImage, vertices)

    # Perspective transform
    src = np.float32([[598,450],[682,450],[210,imshape[0]],[1110,imshape[0]]])
    dst = np.float32([[350,0],[950,0],[350,imshape[0]],[950,imshape[0]]])
    warpedImage, M, Minv = warp(maskedImage,src, dst)

    # Return out_img_before_imvwarp for test
    tmp_image, offset, left_curverad, right_curverad, gap, out_img_before_invwarp = fit_polynomial(warpedImage, Minv)
    result = cv2.addWeighted(image, 1, tmp_image, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,"Radius of Curvature = "+str('%.2f'%((left_curverad+right_curverad)/2))+"(m)",(50,50), font, 0.7,(255,255,255),1,cv2.LINE_AA)
    
    np.average(np.nonzero(tmp_image[:,:,1]))

    if offset > 0:
        position = 'right '
    if offset == 0:
        position = ' '
    if offset < 0:
        offset = -1*offset
        position = 'left '

    cv2.putText(result,"Vehicle is "+str('%.2f'%offset) + "m " + position + "of center",(50,75), font, 0.7,(255,255,255),1,cv2.LINE_AA)
    
    return result

if __name__ == '__main__':
    # Get camera calibration parameters for this project.
    ret, mtx, dist, rvecs, tvecs = cameraCalibraion(chessBoardImageFiles)

    # Read project video and draw lane lines in it
    clip = VideoFileClip(projectVideo)
    modifiedClip = clip.fl_image(drawLaneLines)

    modifiedClip.write_videofile(output)
    




