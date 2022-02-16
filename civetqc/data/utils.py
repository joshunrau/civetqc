import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import sobel, prewitt_h, prewitt_v
from skimage.feature import canny
from skimage.io import imread, imshow
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import clear_border, watershed

def load(filepath):
    """ return image without alpha channel """
    return imread(filepath)[:,:,:3]

def show(img):
    plt.figure(figsize = (30, 10), dpi = 150)
    imshow(img)

def get_segmentation(img):
    img_ng = np.copy(img)
    img_ng[:,:,1] = np.zeros([img_ng.shape[0], img_ng.shape[1]]) # Remove all green
    img_grey = rescale_intensity(rgb2gray(img_ng))
    elevation_map = sobel(img_grey)
    markers = np.zeros_like(img_grey)
    markers[img_grey < .001] = 1
    markers[img_grey > .15] = 2
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    return segmentation
    
def split_segments(img, segmentation):
    label_image = label(segmentation)
    regions = regionprops(label_image)
    images = []
    for region in regions:
        images.append(img[region.slice])
    return images
    
def split_verify_images(img):
    segmentation = get_segmentation(img)
    return split_segments(img, segmentation)

def get_color_maps(img):
    rgb_maps = [np.copy(img), np.copy(img), np.copy(img)]
    rgb_maps[0][:,:,1], rgb_maps[0][:,:,2] = 0, 0
    rgb_maps[1][:,:,0], rgb_maps[1][:,:,2] = 0, 0
    rgb_maps[2][:,:,0], rgb_maps[2][:,:,1] = 0, 0
    return [rescale_intensity(rgb2gray(color)) for color in rgb_maps]

def from_color_maps(red, green, blue):
    return np.stack((red, green, blue), axis = -1)

def get_pure_color_maps(img):
    red, green, blue = get_color_maps(img)
    pure_red = np.logical_and.reduce([red == 1, green == 0,  blue == 0])
    pure_green = np.logical_and.reduce([red == 0, green == 1, blue == 0])
    pure_blue = np.logical_and.reduce([red == 0, green == 0, blue == 1])
    return pure_red, pure_green, pure_blue