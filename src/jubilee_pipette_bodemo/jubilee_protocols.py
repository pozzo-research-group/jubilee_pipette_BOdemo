"""
This module should handle all interactions with the real world
"""
import image_processing as img
import numpy as np
from datetime import date



def sample_point(jubilee, pipette, Camera, color_composition: tuple, volume: float, well, color_stocks, save =True):
    """
    Sample a specified point. 

    Inputs:
    jubilee: Jubilee Machine object
    Pipette: Jubilee library pipette object, configured
    Camera: Jubilee library camera tool
    color_composition (tuple) - (R, Y, B) values - either 0-1 or 0-255.
    volume: total sample volume
    well: location to prepare sample
    red_stock: location for red stock 
    yellow_stock: location for yellow stock
    blue_stock: location for blue stock

    Returns:
    -------
    RGB - tuple RGB value of resulting solution
    """

    volumes = color_composition*volume
    print('Calculated volumes: ', volumes)

    # Let's check so that we can mix after the last color was added
    for i, v in enumerate(volumes):
        if v != 0:
            stock_to_mix = i

    # pipette colors into well
    jubilee.pickup_tool(pipette)


    pipette.transfer(volumes, color_stocks, well.top(-1), 
                     blowout = True, new_tip='once', mix_after = (275, 5, color_stocks[stock_to_mix]))

    jubilee.pickup_tool(Camera)

    image = Camera.capture_image(well, light= True, light_intensity=1)

    jubilee.park_tool()

    RGB = img.process_image(image)

    if save:
        td = date.today().strftime("%Y%m%d")
        filename = f"{td}_{well.name}_{well.slot}"
        img.save_image(image, './', filename)

    return RGB, image

