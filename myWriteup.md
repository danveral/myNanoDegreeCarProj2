## My Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./output_images/undistort_image.png "undistort"
[image2]: ./output_images/test2.jpg "origin"
[image2_2]: ./output_images/undistort_test2.png "Road Transformed"
[image3]: ./output_images/binary_combo.png "Binary Example"
[image3_2]: ./output_images/masked_image.png "Binary Example"
[image4]: ./output_images/warped_img.png "Warped image"
[image5_1]: ./output_images/histogram.png "Fit Visual"
[image5]: ./output_images/fit_lines2.png "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./output_images/myOutput.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### Computed the camera matrix and distortion coefficients.

The code for this step is contained a function named "cameraCalibraion" which used "chessBoardImageFiles" and return 5 coeffieients --- ret, mtx, dist, rvecs, tvecs.  Then I used mtx and dist to undistort every image I got from the vedio by using "cv2.undistort" in the function "drawLaneLines": 

About the function of "cameraCalibraion", I defined two list: objpoints and imgpoints, by finding the chessboard corners, I can get full objpoints and imgpoints, then I used cv2.calibrateCamera to get those 5 coefficients.

I undistorted the "calibration1.jpg" for example, by using the coefficients dist and mtx, here is the comparation images before and after undistortion:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

This is the orignal image:
![alt text][image2]

Then I undistort this image by using "cv2.undistort" method.
![alt text][image2_2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (Please refer to "colorAndGradient(undistortImage)" method in "myCode.py".  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Before the perspective transform, I applied a mask of this image, it's not necessary, but I believe it can make output better.
This mask is defined in the function of "regionOfInterest"
The image applied by mask is like this:
![alt text][image3_2]


The code for my perspective transform includes a function called `warp()`, which appears in the file `myCode.py` The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[598,450],[682,450],[210,imshape[0]],[1110,imshape[0]]])
dst = np.float32([[350,0],[950,0],[350,imshape[0]],[950,imshape[0]]])
```

Those are:

| Source        | Destination   |
|:-------------:|:-------------:|
| 598, 450      | 350, 0        |
| 682, 450      | 950, 0      |
| 210, 720     | 350, 720      |
| 1110, 720      | 950, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used histogram method to find the lane line points, I tried other methods I learned, but I found the histogram is the best.

![alt text][image5_1]

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function "fit_polynomial" in my code in `myCode.py`
when I got the left line and right line points (leftx, lefty, rightx, righty) in the function "find_lane_pixels", I used np.polyfit method to find out the left and right curve's coefficient(2nd order polynomial).

I made a filter that to make sure left and right lines are approximate parallel, if not, use the previous frame's lines. also, I measured in x-dimension, 3.7m represents 600 pixels, and in y-dimension, 30m represents 780 pixels, so I define variables like this:

    xm_per_pix = 3.7/600
    ym_per_pix = 30./780

then changed pixels to meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in my code in `myCode.py` in the function `drawLaneLines()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/myOutput.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First, I used a combination of color and gradient thresholds, I just felt they are not robust enough. if the car running in some heavy tree shadow, it may fail to find out the lane lines. Also I used a mask to get lane line detection better, but if the lane line is too curve, it also may not work.

so I think the methods I choose is very beginning of the lane lines detection, but I think I can keep learning.
