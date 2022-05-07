import numpy as np
import cv2
import os
from scipy.signal import convolve2d

def apply_dewarping(img, warped_corners, actual_corners):
    transform_img = cv2.getPerspectiveTransform(warped_corners, actual_corners) # get required transform of image
    dewarped_img = cv2.warpPerspective(img, transform_img, (np.shape(img)[1], np.shape(img)[0])) # dewarp image using transform
    return dewarped_img

def apply_noise_filtering(img):
    median_img = cv2.medianBlur(img, 3) # apply 3 x 3 mean filter to img    
    nl_means_denoised_img = cv2.fastNlMeansDenoising(median_img, None, 3) # apply 3 x 3 non-local means filter to median filtered image
    return nl_means_denoised_img

def apply_contrast_and_brightness_adjustment(img):
    img = np.array(img, dtype=np.int32) # change dtype of array so it can go below 0 and above 255 (avoids integer overflow)
    c = 255 / np.log(1 + np.max(img)) # scaling factor for log transform
    img = c * np.log(img + 1) # perform log transform
    img = (1.0175 ** img - 1) # exponential transform
    img = 255 * (1 - (np.max(img) - img) / np.max(img)) # scale img back to 0-255 range
    return np.array(img ** 0.95, dtype=np.uint8) # return image with correct dtype, and after performing gamma-correction

in_directory = "" # Directory to read images in from
out_directory = "Results" # Directory to write processed images to

test_image = cv2.imread(in_directory + "/" + os.listdir(in_directory)[0], cv2.IMREAD_GRAYSCALE) # Read first image from in_directory

height = np.shape(test_image)[0] # Height of test image
width = np.shape(test_image)[1] # Width of test image
y_nonzero, x_nonzero = np.where(test_image != 0) # Find all x and y coordinates where pixel value is non-zero

# We know all images are warped in the same way, so we can just perform operations on the test image to find the perspective transform.

# By observing the test image, the y-coordinate for the bottom left warped corner will be the maximum y-value for which the pixel value
# is non-zero, then the corresponding x-coordinate is the minimum x-value for which the pixel value is non-zero at that y-value.
y_bl = np.max(y_nonzero) # y-coordinate of bottom left warped corner
x_bl = np.min(np.where(test_image[y_bl, :] != 0)) # x-coordinate of bottom left warped corner

# Similar process to before, only that we first find x-coordinate, since the bottom right warped corner will be the maximum x-value
# for which the pixel value is non-zero, then corresponding y-coordinate is the maximum y-value for which the pixel is non-zero
# at that x-coordinate
x_br = np.max(x_nonzero) # x-coordinate of bottom right warped corner
y_br = np.max(np.where(test_image[:, x_br] != 0)) # y-coordinate of bottom right corner

# Same idea as with bottom left
y_tr = np.min(y_nonzero) # y-coordinate of top right warped corner
x_tr = np.max(np.where(test_image[y_tr, :] != 0)) # x-coordinate of top right warped corner

# Can't apply same procedure for top left, so instead we start from the top left of the entire image, and gradually increase an n x n matrix
# to find the first time a pixel value is non-zero
i = 0
while np.sum(test_image[0:i + 1, 0:i + 1]) == 0:
    i += 1
    
#print(test_image[0:i+1, 0:i+1]) # We see that the first non-zero element is at the very bottom right of the (i+1) x (i+1) matrix, so y_tl = i
y_tl = i
x_tl = np.min(np.where(test_image[y_tl, :] != 0))

warped_corners = np.float32([[x_tl, y_tl], [x_tr, y_tr], [x_bl, y_bl], [x_br, y_br]]) # Corners of the warped image
actual_corners = np.float32([[0, 0], [width, 0], [0, height], [width, height]]) # Corresponding corners we want to transform to

frameSize = (width, height) # size of frame for image
output_video_filename = "Result.avi" # path to result video
out = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc(*'DIVX'), 3, frameSize, isColor=False) # create video writer object

# create sharpening kernel
sharpen_kernel = np.zeros((3, 3))
sharpen_kernel[0][1] = -1
sharpen_kernel[1][0] = -1
sharpen_kernel[1][1] = 5
sharpen_kernel[1][2] = -1
sharpen_kernel[2][1] = -1

try:
    os.mkdir(out_directory) # try to create directory specified by out_directory
except:
    None # if it already exists, continue

for img_filename in os.listdir(in_directory):
    img = cv2.imread(in_directory + "/" + img_filename, cv2.IMREAD_GRAYSCALE) # read in current image
    if img is not None:    
        noise_removed_img = apply_noise_filtering(img) # filter out noise from the image
        dewarped_img = apply_dewarping(noise_removed_img, warped_corners, actual_corners) # dewarp the image
        contrast_enhanced_img = apply_contrast_and_brightness_adjustment(dewarped_img) # apply contrast and brightness enhancement
        sharp_img = cv2.resize(convolve2d(contrast_enhanced_img, sharpen_kernel), (width, height)) # sharpen image (resize since result is wrong size)
        cv2.imwrite(out_directory + "/" + img_filename, sharp_img) # write image to out_directory
        final_img = cv2.imread(out_directory + "/" + img_filename, cv2.IMREAD_GRAYSCALE) # read image back in again (get an error if we try to add the sharp_img to the video)
        out.write(final_img) # write final_img to video       
    else:
        print("Error loading " + in_directory + "/" + img_filename)

out.release() # write video to file