import numpy as np
import image_processing as img
from bayesopt import bayesian_optimzer, acquisitions
import jubilee_protocols


"""
This module should handle the bayesian optimization loop type stuff
"""

def BO_campaign(initial_data, acquisition_function, number_of_iterations, jubilee, pipette, camera, sample_volume, red_stock, yellow_stock, blue_stock, samples):
    """
    This should be a child-safed way to run BO on the platform

    initial data: The already acquired initial data points to seed BO
    acquisition function:
    number of iterations: number of samples to test in BO campaign
    jubilee: jubilee object
    pipette: jubilee pipette object
    camera: jubilee camera object
    sample_volume: volume to make for sample, probably in mL but depends on pipette
    red_stock, yellow, blue stock: wells with stock location
    samples: labware object to make samples in. For now this only works with one piece of labware.    
    """
    n_points = 101 # number of grid points on sampling grid

    # define possible sampling grid
    available_points = get_constrained_points(n_points)
    # we know we are working with a 3-variable constrained design space here so can just hard-code that
    # instantiate a bayesian optimizer object
    bo = bayesian_optimzer.BayesianOptimizer(None, acquisition_function, None, initial_data)

    # check that we have enough sample wells for the number of iterations we want to run 
    assert len(samples.wells < number_of_iterations), 'Error: Too many samples to test for number of wells in labware.'

    # get first set of points from model
    query_point = bo.campaign_iteration(None, None)

    for i in range(number_of_iterations):
        # query point from BO model
        # get well
        print(f'Starting iteration {i}')

        # figure out how to get the next well with the new setup
        well = samples[i]
        # run point in real world
        print(f'Dispensing into well {well}')
        print('RYB values tested: {query_point}')
        new_color = jubilee_protocols.sample_point(jubilee, pipette, camera, query_point, sample_volume, well, red_stock, yellow_stock, blue_stock)

        print('RGB values observed: {RGB}')
        query_point = bo.campaign_iteration(query_point, new_color)




def get_constrained_points(n_points):
    """
    Get the available points in 3-dimensional sample space with volumetric mixing constraint
    """


    R = np.linspace(0, 1, n_points)
    Y = np.linspace(0, 1, n_points)
    B = np.linspace(0,1, n_points)

    # do a brute force constrained sampling to get points in the design space
    test_arr = np.array(np.meshgrid(R, Y, B)).T.reshape(-1,3)
    indices = np.where(test_arr.sum(axis = 1) != 1)[0]
    testable_points = np.delete(test_arr, indices, axis = 0)

    return testable_points

def initial_random_sample(testable_points, n_sample = 12):
    """
    Sample n_sample points from testable_points to get initial data
    """
    rng = np.random.default_rng(seed = 4)
    selected_inds = rng.integers(0, len(testable_points), n_sample)
    selected_points = testable_points[selected_inds, :]
   
    return selected_points