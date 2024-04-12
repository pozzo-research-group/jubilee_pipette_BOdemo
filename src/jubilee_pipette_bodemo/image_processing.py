import cv2
import numpy as np

def process_image(image_bin):
    """
    externally callable function to run processing pipeline
    
    takes an image as a bstring
    """
    image_arr = np.frombuffer(image_bin, np.uint8)
    image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    radius = 30
    masked_image = _mask_image(image, radius)
    values = _get_rgb_avg(masked_image)
    return values

def _mask_image(image, radius):

    mask = np.zeros(image.shape[:2], dtype = "uint8")
    cv2.circle(mask, (300, 300), radius, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)

    return masked

def _get_rgb_avg(image):
    bgr = []
    for dim in [0,1,2]:
        flatdim = image[:,:,dim].flatten()
        indices = flatdim.nonzero()[0]
        value = flatdim.flatten()[indices].mean()
        bgr.append(value)

    rgb = [bgr[i] for i in [2,1,0]]

    return rgb