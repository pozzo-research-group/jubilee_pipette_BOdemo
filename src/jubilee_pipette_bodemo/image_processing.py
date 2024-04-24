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

def _mask_image(image, radius= 50):
    """Apply a circular mask to an image

    :param image: the image object
    :type image: np.array
    :param radius: the size (in pixels) of the circular mask, defaults to 50
    :type radius: int, optional
    :return: the masked image
    :rtype: np.array
    """
    image_shape = image.shape[:2]
    w = image_shape[0]//2
    h = image_shape[1]//2
    mask = np.zeros(image_shape, dtype = "uint8")
    cv2.circle(mask, (w, h), radius, 255, -1)
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


def save_image(image_bin, filename):
    image_arr = np.frombuffer(image_bin, np.uint8)
    image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    cv2.imwrite(f'./{filename}.jpg', image)
    return