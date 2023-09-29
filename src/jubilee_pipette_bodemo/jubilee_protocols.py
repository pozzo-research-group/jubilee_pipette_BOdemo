"""
This module should handle all interactions with the real world
"""
import image_processing as img
import numpy as np



def sample_point(jubilee, pipette, Camera, RYB: tuple, volume: float, well, red_stock, yellow_stock, blue_stock, trash_well):
    """
    Sample a specified point. 

    Inputs:
    jubilee: Jubilee Machine object
    Pipette: Jubilee library pipette object, configured
    Camera: Jubilee library camera tool
    RYB (tuple) - (R, Y, B) values - either 0-1 or 0-255.
    volume: total sample volume
    well: location to prepare sample
    red_stock: location for red stock 
    yellow_stock: location for yellow stock
    blue_stock: location for blue stock

    Returns:
    -------
    RGB - tuple RGB value of resulting solution
    """

    ########
    # Volume calculation code
    #################
    RYB = list(RYB)
    # get the volumes of each color
    if np.isclose(sum(RYB), 1):
        pass
    elif np.isclose(sum(RYB) ,255):
        RYB = [i/255 for i in RYB]
    else:
        raise AssertionError('Error: Volume fractions of RYB must add to 1 or 255')
    
    volumes = [vf*volume for vf in RYB]    
    

    ###############
    # Liquid handling stuff
    ###############

    # pipette colors into well
    # pipette red:
    #pickup the red tip from rack
    jubilee.pickup_tool(pipette)
    #this is a mess to make sure we aren't pipetting more than 300 ul. Fix in transfer function

    if volumes[0] > 300:
        try:
            red_tip = pipette.red_tip
        except AttributeError:
            red_tip = None

        pipette.pickup_tip(red_tip)

        if red_tip is None:
            pipette.red_tip = pipette.first_available_tip
        #aspirate v[0] of red solution
        pipette.transfer(volumes[0]/2, red_stock, well)
        pipette.transfer(volumes[0]/2, red_stock, well)
        # return tip to same location
        pipette.return_tip()
    else:
        try:
            red_tip = pipette.red_tip
        except AttributeError:
            red_tip = None

        pipette.pickup_tip(red_tip)

        if red_tip is None:
            pipette.red_tip = pipette.first_available_tip
        #aspirate v[0] of red solution
        pipette.transfer(volumes[0], red_stock, well)
        # return tip to same location
        pipette.return_tip()

    # same for yellow
    if volumes[1] > 300:
        try:
            yellow_tip = pipette.yellow_tip
        except AttributeError:
            yellow_tip = None

        pipette.pickup_tip(yellow_tip)

        if yellow_tip is None:
            pipette.yellow_tip = pipette.first_available_tip

        pipette.transfer(volumes[1]/2, yellow_stock, well)
        pipette.transfer(volumes[1]/2, yellow_stock, well)
        # return tip to same location
        pipette.return_tip()
    else:
        try:
            yellow_tip = pipette.yellow_tip
        except AttributeError:
            yellow_tip = None

        pipette.pickup_tip(yellow_tip)

        if yellow_tip is None:
            pipette.yellow_tip = pipette.first_available_tip

        pipette.transfer(volumes[1], yellow_stock, well)
        # return tip to same location
        pipette.return_tip()

    # for blue:
    # get a new tip
    if volumes[2] > 300:
        pipette.pickup_tip()
        # aspirate, dispense
        pipette.transfer(volumes[2]/2, blue_stock, well)
        pipette.transfer(volumes[2]/2, blue_stock, well, mix_after = (300, 3))
        # discard tip 
        pipette.drop_tip(trash_well)
    else:
        pipette.pickup_tip()
        # aspirate, dispense
        pipette.transfer(volumes[2], blue_stock, well, mix_after = (300, 3))
        # discard tip 
        pipette.drop_tip(trash_well)


    jubilee.park_tool(pipette)
    
    # will this park the tool automatically?
    jubilee.pickup_tool(Camera)
    image = jubilee.well_image(well)
    image = Camera.capture_image(well)
    # do post-processing 
    RGB = img.process_image(image)
    
    return RGB