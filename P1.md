# **Finding Lane Lines on the Road**

## Self-Driving Car Nanodegree Term 1 Project 1
### Scott Henderson

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image0]: ./test_images/solidWhiteRight.jpg "Original Image"
[image1]: ./test_images_output/solidWhiteRight_1Gray.jpg "Grayscale"
[image2]: ./test_images_output/solidWhiteRight_2Blur.jpg "Blurred Grayscale"
[image3]: ./test_images_output/solidWhiteRight_3CannyEdges.jpg "Canny Edges"
[image4]: ./test_images_output/solidWhiteRight_4MaskedCannyEdges.jpg "Masked Canny Edges"
[image5]: ./test_images_output/solidWhiteRight_5HoughLines.jpg "Hough Lines"
[image6]: ./test_images_output/solidWhiteRight_LaneLines.jpg "Lane Lines"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My image pipeline consisted of 5 major steps.

Original image:
![alt text][image0]

1. Convert the original color image to grayscale.
![alt text][image1]

1. Apply a Gaussian blur to the grayscale image. A simple Gaussian blur is included in the OpenCV Canny edge detection function but the caller does not have control over the details. So apply a Gaussian blur outside the Canny edge detection function. This allows us to control the kernel size and try different values to see which may work best.
![alt text][image2]

1. Use Canny edge detection on the blurred image to find edges. The Canny edge detection algorithm uses two threshold parameters: low and high. A pixel is classified according to it's gradient value. Since we're dealing with an 8-bit grayscale image this leads to "color" values in the range [1, 2^8] or [1, 256].

  If a pixel's value is greater than the high threshold then the pixel is considered a strong edge pixel
  If a pixel's value is between the low threshold and high threshold is it considered to be a weak edge pixel
  If a piexl's value is below the low threshold it is ignored entirely.

  Canny recommends a ratio of 1:2 or 1:3 for low:high threshold.
  ![alt text][image3]

1. Mask the resulting Canny edge image so that we're looking at just the edges near what we believe to be the lane. We are not interested in other objects along the sides of the road or other lanes or other objects not on the road at this time. The mask process consists of creating a polygon (really just a quadrilateral in this case) and then applying this to mask the Canny edge image.
![alt text][image4]

1. Finally we apply a Hough transform to the clipped Canny edges to find lane lines. The Hough transform algorithm uses a few parameters:

  (rho, theta) - These two parameters describe a grid - in polar coordinates - to be applied to the Canny edge image. The grid size of each grid cell is rho while theta determines the angular resolution (in radians) of the grid.

  threshhold - This value is the minimum number of votes (intersections in a Hough grid cell) are required for an edge to be considered part of a line.

  min. line length - This value is the minimum length (in pixels) of a line. Lines shorter than this value will not be generated.

  max. line gap - This value is the maximum gap (in pixels) between edges. If the gap length is less than this value then two edges will be joined to form a single line. Otherwise the gap remains and we'll end up with two lines.
  ![alt text][image5]

Final image consisting of the original image with Hough lines overlaid:
![alt text][image6]


In order to draw a single line on the left and right lanes, I modified the draw_lines() function.  An initial loop does two things:

1. Calculate the slope and center point for each line segment.
1. Partition the line segments into 'left' and 'right' lists using the slope. If the slope is less than zero we assume that the line segment belongs to the left side lane line. If the slope is greater than zero we assume that the line segment belongs to the right side lane line. If the slope is zero or greater than an arbitrary epsilon value (set to 0.9 in my code) then the line segments is ignored. This eliminates problems with vertical or horizontal lines.

The next step is to calculate the average slope and average center point for all line segments (still separated into left and right lists).  With these values we can use the simple slope-intercept line formula:

    y = mx + b

To generate points on an 'average' line representing all line segments for the left and right, respectively. For simplicity we ignore the y-intercept value (b) and derive new points like so:

    y = mx + b
    (y - y') = m(x - x')
    (y - y') / m = x - x'
    x' = x - ((y - y') / m)

where

    (x, y) is the calculated average center point
    m is the calculated average slope
    (x', y') is the point on our representative line

To obtain lines that stretch from the bottom of our image to the 'lane line horizon' near the top of our image we plug two y values into this formula (recall that the origin is in the upper left corner of the image so the max y coordinate value is the bottom of the image):

    y' = image.shape[0]        # max. y coordinate value for the image
    y' = 0.6 * image.shape[0]  # estimate for the top of our lane line horizon



### 2. Identify potential shortcomings with your current pipeline


I noticed when processing the 'solidYellowLeft' sample video, that the dashed white lane lines on the right are drawn as individual line segments closer to the vehicle and as a solid line farther away. It might be possible to connect the line segments in the foreground by adjusting the max. line gap parameter for the Hough transform. Though this could also lead to other undesireable artifacts.

The modified version of draw_lines() is another way to "connect the line segments" for the lane lines but there are potential problems with this solution. At least two that occur to me but both are related to lane lines that curve more or less sharply.

1. The separation of line segments into left and right lists relies on lane lines sloping towards the center of the image from the left and right side, respectively. However, for sharp corners this partition method may break down as the slope becomes too close to horizontal.
1. Assuming we can resolve the 'left' vs. 'right' issue we are still left with a lane line that may not be particularly straight. Since we're using just a simple line formula the line may not match the curve of the lane line. It's possible that the lane line will be close enough closer to the vehicle but the accuracy could diverge significantly farther away from the vehicle due to the curing lane line.

I attempted to fit a curve using a polynomial with all of the line segment points but the results were somewhat less than impressive.


### 3. Suggest possible improvements to your pipeline

The first thing I would note is that the various parameters used for the Gaussian blur, Canny edge detection and Hough transform steps are all arrived at by trial and error. It seems that more investigation could be done to refine these parameters.

I've also noticed that the left and right lane line segments sometimes converge in the center of the lane. This behavior appeared after I modified the draw_lines() function so there's probably some residual issue with that function. In fact the lines drawn for the 'solidYellowLeft' sample video leave much to be desired with the left and right lines frequently crossing or going off into obviously non-lane space.

Another problem I ran into in initial testing on the 'solidYellowLeft' sample video is that there are cases where there are no left lines or no right lines due to the simple method used to partition lines into left and right. Definitely some improvement needed in this area.

The issues noted above for the 'solidYellowLeft' sample video were amplified many times in the 'challenge' video sample. But the problems with the 'challenge' video go much deeper. If I revert to the original, simple version of draw_lines() that does not attempt to smooth line segments in any way but just draws them as is, I can see a lot of extra lines in the output video. This tells me that there is something fundamentally wrong with my image pipeline for processing images from the 'challenge' video. It may be as simple as removing the vehicle hood from the sample images and/or adjusting the Canny and/or Hough parameters but this assignment is already a week late so ...
