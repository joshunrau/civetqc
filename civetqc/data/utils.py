import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import sobel, prewitt_h, prewitt_v
from skimage.feature import canny
from skimage.io import imread, imshow
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, watershed


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
    for i in range(3):
        for j in range(3):
            if i != j:
                rgb_maps[i][:,:,j] = 0
    return [rescale_intensity(rgb2gray(color)) for color in rgb_maps]