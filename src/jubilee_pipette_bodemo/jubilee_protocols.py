"""
This module should handle all interactions with the real world
"""




def sample_point(jubilee, pipette, Camera, RYB: tuple, volume: float, well, red_stock, yellow_stock, blue_stock):
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
    RYB = list(RYB)
    # get the volumes of each color
    if np.isclose(sum(RYB), 1):
        pass
    elif np.isclose(sum(RYB) ,255):
        RYB = [i/255 for i in RYB]
    else:
        raise AssertionError('Error: Volume fractions of RYB must add to 1 or 255')
    
    volumes = [vf*volume for vf in RYB]    
    


    # pipette colors into well
    # pipette red:
    #pickup the red tip from rack

    #aspirate v[0] of red solution

    # dispense into well

    # return tip to same location


    # same for yellow

    # for blue:
    # get a new tip

    # aspirate, dispense

    #mix

    # discard tip 



    pipette.transfer(volumes[0], source_well = red_stock, destination_well = well, 
                     )
    
    # will this park the tool automatically?
    jubilee.pickup_tool(Camera)
    image = jubilee.well_image(well)
    image = Camera.capture_image(well)
    # do post-processing 
    RGB = img.process_image(image)
    
    return RGB