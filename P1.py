# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 06:56:49 2018

@author: Scott

Self-Driving Car Engineer Nanodegree

Project: Finding Lane Lines on the Road

"""

import os
import sys
import logging
import datetime

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2  # OpenCV

from moviepy.editor import VideoFileClip


def init(log_file_base='LogFile', logging_level=logging.INFO):
    """
    Application initialization.

    Set up a log file object.

    Args:
        log_file_base (str):    Base log file name (date will be appended).
        log_level (Level):      Log message level

    Returns:
        Logger: Object to be used for logging
    """
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    string_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = '{}_{}.log'.format(log_file_base, string_date)
    log = setup_logger(log_dir, log_file, log_level=logging_level)

    return log


# Setup Logging
def setup_logger(log_dir=None,
                 log_file=None,
                 log_format=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                              datefmt="%Y-%m-%d %H:%M:%S"),
                 log_level=logging.INFO):
    """
    Setup a logger.

    Args:
        log_dir (str):          Log file directory
        log_file (str):         Log file name
        log_format (Formatter): Log file message format
        log_level (Level):      Log message level

    Returns:
        Logger: Object to be used for logging
    """
    # Get logger
    logger = logging.getLogger('')
    # Clear logger
    logger.handlers = []
    # Set level
    logger.setLevel(log_level)
    # Setup screen logging (standard out)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(log_format)
    logger.addHandler(sh)
    # Setup file logging
    if log_dir and log_file:
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        fh.setFormatter(log_format)
        logger.addHandler(fh)

    return logger


def load_image(image_file, log=None):
    """
    Load an image.

    Args:
        image_file (str):   Image file name

    Returns:
        image: The loaded image object.
    """
    if log:
        log.info('Loading image ''{}'' ...'.format(image_file))
    else:
        print('Loading image ''{}'' ...'.format(image_file))
    image = mpimg.imread(image_file)
    if log:
        log.debug('Image {0}:\ntype: {1}\ndimensions: {2}'.format(
                  image_file, type(image), image.shape))

    return image


def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel

    Args:
        img (image):        Blur this image
        kernel_size (int):  Use this kernel size for blurring the image.

    Returns:
        image: The blurred image.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """
    Applies the Canny transform

    Args:
        img (image):            Transform this image
        low_threshold (int):    Canny low threshold
        high_threshold (int):   Canny high threshold

        If an edge pixel's gradient value is:
            > high_threshold
                it is marked as a strong edge pixel
            in [low_threshold, high_threshold]
                it is marked as a weak edge pixel
            < low_threshold
                it will be suppressed

        For a grayscale image we have 8 color bits == 2^8 == 256 so the
        thresholds in this case would be in [1, 256]

        Canny recommends a ratio of 1:2 or 1:3 for low-to-high threshold.

    Returns:
        image: The result of applying Canny edge detection.
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def mask_polygon(img, top, delta_top, delta_bottom):
    """
    Given an image, return a list of vertices describing a quadrilateral
    that can be used to mask the image for lane line detection.

    Args:
        img (image):        The image to be masked (we need this only for shape).
        top (int):          The top of the mask polygon.
        delta_top (int):    Half of the width of the top of the mask polygon.
        delta_bottom (int): Half of the width of the bottom of the mask polygon.

    Returns:
        numpy array of vertices: A set of four vertices describing a quadrilateral.
    """
    imshape = img.shape  # 0 == y, 1 == x
    lower_left  = (delta_bottom,               imshape[0])
    upper_left  = (imshape[1] / 2 - delta_top, top)
    upper_right = (imshape[1] / 2 + delta_top, top)
    lower_right = (imshape[1] - delta_bottom,  imshape[0])
    vertices = np.array([[lower_left, upper_left, upper_right, lower_right]],
                        dtype=np.int32)
    return vertices


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.

    Args:
        img (image):                        Mask this image
        vertices (numpy array of integers): A polygon to use for the mask

    Returns:
        image: The masked image.
    """
    # Define a blank mask to start with
    mask = np.zeros_like(img)

    # Define a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap,
                log=None):
    """
    Apply the Hough transform

    Args:
        img (image):        Transform this image - should be output of Canny transform
        rho (int):          Distance resolution in pixels of the Hough grid.
        theta (int):        Angular resolution in radians of the Hough grid.
        threshold (int):    Min. number of votes (intersections in a Hough grid cell).
        min_line_len (int): Min. # of pixels making up a line.
        max_line_gap (int): Max. # of piexls between connectable line segments.

    Returns:
        image: image with Hough lines drawn.
    """
    # Generate Hough lines.
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # Create a blank image to draw on.
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Draw lines
    draw_lines(line_img, lines, drawAllLines=True, log=log)
    return line_img


def draw_lines(img, lines, drawAllLines=False, color=[255, 0, 0], thickness=2,
               log=None):
    """
    NOTE: this is the function you might want to use as a starting point once
    you want to average/extrapolate the line segments you detect to map out
    the full extent of the lane (going from the result shown in raw-lines
    example.mp4 to that shown in P1_example.mp4).

    Think about things like separating line segments by their slope
        dy / dx == ((y2 - y1) / (x2 - x1))
    to decide which segments are part of the left line vs. the right line.
    Then, you can average the position of each of the lines and extrapolate
    to the top and bottom of the lane.

    This function draws 'lines' with 'color' and 'thickness'.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below.

    Args:
        img (image):                Draw lines on this image
        lines (array of lines as pairs of integer coords):
                                    Draw these lines
        drawAllLines (bool):        Draw all line segments or attempt to average?
        color (RGB array of ints):  Draw lines using this color.
        thickness (int):            Draw lines using this thickness

    Returns:
        None
    """
    if drawAllLines:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        epsilon = 0.9  # Threshold to test for valid slope.

        # Calculate the average slope and center point for each line segment.
        # Then use the slope-intercept line formula to generate end points for
        # a straight line through the average center with the average slope.
        # This will not work at all well for curves but is adequate for
        # relatively straight lane lines.

        # Calculate slope and center for each line segment. Use the slope to
        # separate lines into left and right.
        lm = []
        rm = []
        lc = []
        rc = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                m = ((y2 - y1) / (x2 - x1))         # slope (dy / dx)
                c = [(x2 + x1) / 2, (y2 + y1) / 2]  # center
                if log:
                    log.debug('slope: {0}'.format(m))
                if -epsilon < m and m < 0.:     # negative slope == left
                    lm.append(m)
                    lc.append(c)
                elif 0. < m and m < epsilon:    # positive slope == right
                    rm.append(m)
                    rc.append(c)

        # Take the average slope and average center so we have single slope and
        # center values for left and right.
        l_slope = np.average(lm)
        r_slope = np.average(rm)

        l_center = np.average(lc, axis=0)
        r_center = np.average(rc, axis=0)

        if log:
            log.debug('r/l slope={0}/{1}'.format(l_slope, r_slope))
            log.debug('r/l center={0}/{1}'.format(l_center, r_center))

        # Start with the standard slope-intercept line formula:
        # y = mx + b
        # Ignore the y-intercept value (b).
        # (y - y') = m(x - x')
        # (y - y') / m = x - x'
        # x' = x - ((y - y') / m)

        # Use the center and slope calculated above as our (x, y) and m values,
        # respectively.
        # For y' we'll use the bottom of the image for one point and the mid-point
        # of the image for our other point.
        imshape = img.shape  # 0 == y, 1 == x
        y1 = int(imshape[0])
        y2 = int(0.6 * imshape[0])

        # Left lane line
        if len(lc) > 0:
            x1 = int(l_center[0] - ((l_center[1] - y1) / l_slope))
            x2 = int(l_center[0] - ((l_center[1] - y2) / l_slope))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        # Right lane line
        if len(rc) > 0:
            x1 = int(r_center[0] - ((r_center[1] - y1) / r_slope))
            x2 = int(r_center[0] - ((r_center[1] - y2) / r_slope))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_curves(img, lines, color=[255, 0, 0], thickness=2, log=None):
    """
    NOTE: this is the function you might want to use as a starting point once
    you want to average/extrapolate the line segments you detect to map out
    the full extent of the lane (going from the result shown in raw-lines
    example.mp4 to that shown in P1_example.mp4).

    Think about things like separating line segments by their slope
        dy / dx == ((y2 - y1) / (x2 - x1))
    to decide which segments are part of the left line vs. the right line.
    Then, you can average the position of each of the lines and extrapolate
    to the top and bottom of the lane.

    This function draws 'lines' with 'color' and 'thickness'.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below.

    Args:
        img (image):                Draw lines on this image
        lines (array of lines as pairs of integer coords):
                                    Draw these lines
        color (RGB array of ints):  Draw lines using this color.
        thickness (int):            Draw lines using this thickness

    Returns:
        None
    """
    epsilon = 0.9  # Threshold to test for valid slope.

    # Separate points into left and right lane lines using slope.
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            m = ((y2 - y1) / (x2 - x1))  # slope (dy / dx)
            if log:
                log.debug('slope: {0}'.format(m))
            if -epsilon < m and m < 0.:     # Left lane line
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            elif 0. < m and m < epsilon:    # Right lane line
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    # Use a 3rd degree polynomial for curve fitting.
    degree = 3

    # Draw left lines.
    draw_polylines(img, left_x, left_y, color, thickness, degree, log)

    # Draw right lines.
    draw_polylines(img, right_x, right_y, color, thickness, degree, log)


def draw_polylines(img, x, y, color=[255, 0, 0], thickness=2, degree=3,
                   log=None):
    """
    Draw lines on an image using the specified list of points to generate
    a curve.

    Args:
        img (image):                Draw lines on this image
        x (list of int):            x coordinates of points
        y (list of int):            y coordinates of points
        color (RGB array of ints):  Draw lines using this color.
        thickness (int):            Draw lines using this thickness
    """
    # Fit a polynomial of the specified degree.
    z = np.polyfit(x, y, degree)    # coefficients
    f = np.poly1d(z)                # function

    # Set the list of x values. Evenly space x values between the first
    # and last points in our input values. Bookend these values with a
    # point at the bottom of the image and another point near where we
    # believe the horizon will be - or at least as far as we care about
    # lane lines.
#    imshape = img.shape  # 0 == y, 1 == x
#    y0 = int(imshape[0])
#    x0 = solve_for_x(z, y0)
#    yN = int(0.6 * imshape[0])
#    xN = solve_for_x(z, yN)
#    x_new = np.asarray([x0]) + np.linspace(x[0], x[-1], 20) + np.asarray([xN])

    x_new = np.linspace(x[0], x[-1], 20)

    # Use the polynomial function developed above to find y values for
    # each x value.
    y_new = f(x_new)

    # Draw curves.
    pts = list(zip(x_new, y_new))
    cv2.polylines(img, np.int32([pts]),
                  isClosed=False, color=color, thickness=thickness)


def solve_for_x(poly_coeffs, y):
    """
    Given a list of polygon coefficients we want to solve for y.

        f(x) - y = 0

    Args:
        poly_coeffs (list of float):    Polygon coefficients.
        y:                              Find x such that f(x) = y

    Returns:
        The root of the polygon at y.
    """
    pc = poly_coeffs.copy()
    pc[-1] -= y
    return np.real(np.roots(pc)[0])


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    Combine an initial image with the results of Hough transform.

    The result image is computed as follows:

        initial_img * α + img * β + γ

    NOTE: initial_img and img must be the same shape!

    Args:
        img (image):            Result of Hough transform - an image with lines drawn.
        initial_img (image):    The original image before any processing.
        α (float):              Weight for initial image.
        β (float):              Weight for Hough transform line image.
        γ (float):              Adjustment factor (brightness?)

    Returns:
        image: A combined (weighted) image.
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def find_lane_lines(img, output_dir=None, output_base=None, output_ext=None,
                    log=None):
    """
    Apply Canny edge detection and Hough transform to find lane lines
    in an image.

    Args:
        img (image):        Find lines in this image
        output_dir (str):   Output directory for intermediate image files.
        output_base (str):  Base filename for intermediate image files.
        output_ext (str):   Extension / file type for intermediate image files.

    Returns:
        None
    """
    kernel_size = 5             # kernel size for Gaussian blur
    # Canny edge detection:
    canny_low_threshold  = 50   # low threshold for Canny edge detection
    canny_high_threshold = 150  # high threshold for Canny edge detection
    # Masking:
    mask_top          = 315     # top for the mask polygon
    mask_delta_top    = 10      # half of the top width of the mask polygon
    mask_delta_bottom = 35      # half of the bottom width of the mask polygon
    # Hough transform:
    rho   = 2                   # Hough grid: distance resolution (pixels)
    theta = np.pi / 180         # Hough grid: angular resolution (radians)
    hough_threshold = 15        # min. number of votes for a Hough grid cell
    min_line_length = 80        # min number of pixels making up a line
    max_line_gap    = 40        # max gap (pixels) between connectable line segments

    # Convert the image to gray scale.
    gray = grayscale(img)
    if output_dir:
        output_file = output_base + '_1Gray.' + output_ext
        mpimg.imsave(os.path.join(output_dir, output_file), gray,
                     format=output_ext)
    # Apply a Gaussian blur.
    blur = gaussian_blur(gray, kernel_size)
    if output_dir:
        output_file = output_base + '_2Blur.' + output_ext
        mpimg.imsave(os.path.join(output_dir, output_file), blur,
                     format=output_ext)
    # Use Canny edge detection to find edges.
    edges = canny(blur, canny_low_threshold, canny_high_threshold)
    if output_dir:
        output_file = output_base + '_3CannyEdges.' + output_ext
        mpimg.imsave(os.path.join(output_dir, output_file), edges,
                     format=output_ext)
    # Mask edges that we believe are outside the lane area.
    mask_vertices = mask_polygon(edges, mask_top, mask_delta_top,
                                 mask_delta_bottom)
    masked_image = region_of_interest(edges, mask_vertices)
    if output_dir:
        output_file = output_base + '_4MaskedCannyEdges.' + output_ext
        mpimg.imsave(os.path.join(output_dir, output_file), masked_image,
                     format=output_ext)
    # Use Hough transform to combine Canny edges into lines.
    lines_image = hough_lines(masked_image, rho, theta, hough_threshold,
                              min_line_length, max_line_gap, log)
    if output_dir:
        output_file = output_base + '_5HoughLines.' + output_ext
        mpimg.imsave(os.path.join(output_dir, output_file), lines_image,
                     format=output_ext)
    # Draw the Hough lines on the original image
    combo = weighted_img(lines_image, img)
    return combo


def process_image_files(image_files, output_dir, log=None):
    """
    Find lane lines in each image in a given list.

    Args:
        image_files (list of str):  List of image file names.
        output_dir (str):           Output directory for processed image files.

    Returns:
        None
    """

    image_count = len(image_files)
    if log:
        log.info('Processing {0} image files.'.format(image_count))
    else:
        print('Processing {0} image files.'.format(image_count))
    if image_count > 1:
        _, axes = plt.subplots(image_count, sharex=True,
                               figsize=(20, 10 * image_count))
    for i in range(image_count):
        image_file = image_files[i]
        _, tail = os.path.split(image_file)
        output_base, output_ext = os.path.splitext(tail)
        output_ext = output_ext[1:]  # remove the initial '.'

        # Load the image
        img = load_image(image_file, log)

        # Find lane lines and create a combined image showing lane lines.
        combo = find_lane_lines(img, output_dir, output_base, output_ext, log)

        # Display the lane lines image.
        if image_count > 1:
            axes[i].imshow(combo)
        else:
            plt.imshow(combo)

        # Save the lane lines image to a file.
        output_file = output_base + '_LaneLines.' + output_ext
        mpimg.imsave(os.path.join(output_dir, output_file), combo,
                     format=output_ext)

        # Cleanup
        del img

        # Next image file
        ++i


def process_video_file(input_dir, input_file, output_dir, log=None):
    """
    Apply the 'find_lane_lines' function to each frame in a given video file.
    Save the results to a new video file.

    Args:
        input_dir (str):    Look for the input file here.
        input_file (str):   Process this video file.
        output_dir (str):   Write the modified video file here.

    Returns:
        str: The name fo the file containing the result of applying lane
                lines to the input video.

    To speed up the testing process you may want to try your pipeline on a
    shorter subclip of the video. To do so add

        .subclip(start_second,end_second)

    to the end of the line below, where start_second and end_second are integer
    values representing the start and end of the subclip.
    """
    # Open the video file.
    video_clip = VideoFileClip(os.path.join(input_dir, input_file))
    # For each frame in the video clip, replace the frame image with the
    # result of applying the 'find_lane_lines' function.
    # NOTE: this function expects color images!!
    lane_lines_clip = video_clip.fl_image(find_lane_lines)
    # Save the resulting, modified, video clip to a file.
    lane_lines_output = os.path.join(output_dir, input_file)
    lane_lines_clip.write_videofile(lane_lines_output, audio=False)
    # Cleanup
    video_clip.reader.close()
    video_clip.audio.reader.close_proc()
    del video_clip
    lane_lines_clip.reader.close()
    lane_lines_clip.audio.reader.close_proc()
    del lane_lines_clip
    # Return the output file name.
    return lane_lines_output


def main(name):

    process_video = False

    print('Name: {}'.format(name))
    _, tail = os.path.split(name)
    log_file_base, _ = os.path.splitext(tail)

    image_dir = 'test_images'
    output_image_dir = 'test_images_output'
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    video_dir = 'test_videos'
    output_video_dir = 'test_video_output'
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    # Init
    log = init(log_file_base, logging.INFO)

    # Read and display a sample image
    # image = load_image(image_dir, 'solidWhiteRight.jpg', log)
    # plt.imshow(image)
    # if you wanted to show a single color channel image called 'gray',
    # for example, call as plt.imshow(gray, cmap='gray')

    # Run all test images through our image pipeline.
    image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
    process_image_files(image_files, output_image_dir, log)

    # Test lane finding for video.
    if process_video:
        white_output = process_video_file(video_dir, 'solidWhiteRight.mp4',
                                          output_video_dir, log)
        # yellow_output = process_video_file(video_dir, 'solidYellowLeft.mp4',
        #                                    output_video_dir, log)
        # challenge_output = process_video_file(video_dir, 'challenge.mp4',
        #                                       output_video_dir, log)

    # Close the log file.
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)


if __name__ == '__main__':
    main(*sys.argv)
