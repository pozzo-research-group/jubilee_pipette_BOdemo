"""
This module should handle all interactions with the real world
"""
import image_processing as img
import numpy as np
from datetime import date



def sample_point(jubilee, pipette, Camera, sample_composition: tuple, sample_volume: float,
                 well, color_stocks, save =True):
    """
    Sample a specified point. 

    Inputs:
    jubilee: Jubilee Machine object
    Pipette: Jubilee library pipette object, configured
    Camera: Jubilee library camera tool
    sample_composition (tuple) - stock color values - either 0-1 or 0-sample_volume.
    volume: total sample volume
    well: location to prepare sample
    red_stock: location for red stock 
    yellow_stock: location for yellow stock
    blue_stock: location for blue stock

    Returns:
    -------
    RGB - tuple RGB value of resulting solution
    """
    # Calculate volumes
    if np.round(np.sum(sample_composition)) == 1:
        volumes = [sample_volume * sample for sample in sample_composition]
    elif np.round(np.sum(sample_composition)) == sample_volume:
        volumes = sample_composition
    else:
        print(f'Error: Color composition does not sum to 1 or expected sample volume of {sample_volume}')
    
    print('Calculated volumes: ', volumes)

    # Let's check so that we can mix after the last color was added
    for i, v in enumerate(volumes):
        if v != 0:
            stock_to_mix = i

    # pipette colors into well
    jubilee.pickup_tool(pipette)


    pipette.transfer(volumes, color_stocks, well.top(-1), 
                     blowout = True, new_tip='once', mix_after = (275, 3, color_stocks[stock_to_mix]))

    jubilee.pickup_tool(Camera)

    image = Camera.capture_image(well, light= True, light_intensity=1)

    jubilee.park_tool()

    RGB = img.process_image(image)

    if save:
        td = date.today().strftime("%Y%m%d")
        filename = f"{td}_{well.name}_{well.slot}"
        img.save_image(image, filename)

    return RGB, image

