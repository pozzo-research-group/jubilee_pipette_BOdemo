
import cv2
import jubilee_protocols
import matplotlib.pyplot as plt
import numpy as np
import skopt.acquisition as acquisitions

from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from skopt import gp_minimize, Optimizer



# Workaround: https://github.com/gtaylor/python-colormath/issues/104
def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)


class ColorMatcher:
    def __init__(self, total_stocks, target_color, sample_volume):
        self.nstocks = total_stocks
        self.target_color = target_color
        self.sample_volume = sample_volume
        self.optimal_proportions = None
        self.observed_colors = []
        self.color_composition = []
        self.color_scores = []
        self.images = []
        # Initialize optimizer
        self.optimizer = Optimizer([(0,1)] * self.nstocks, base_estimator = 'GP',
                                   acq_func ='EI',acq_optimizer='sampling', n_initial_points=0) # is it ok to initialize with 0 points?

    def generate_initial_data(self, n_samples, sample_volume):
        # Generate n_samples random color samples presented as proportions of stock colors volumes
        color_samples = np.random.dirichlet(np.ones(self.nstocks), n_samples)
        volumes = [sample_volume * sample for sample in color_samples]
        return volumes

    def inital_data(self, color_volumes:list , color_observations :list ):

        self.color_composition = color_volumes
        self._inital_data = {'volumes':color_volumes, 'colors': color_observations}
        self.x0 = np.array(color_volumes)
        self.y0 = np.array([self.color_score(color) for color in color_observations])

         # assuming base Gaussian process estimator
        self.optimizer.tell(self.x0, self.y0)


    def color_score(self, color):
        # color : list of rgb values of sampled color
        # Convert RGB color to Lab color (CIE L*a*b* color space)
        target_rgb = sRGBColor(*self.target_color, is_upscaled =True if max(self.target_color) > 1 else False)
        color_rgb = sRGBColor(*color, is_upscaled =True if max(color) > 1 else False)
        target_lab = convert_color(target_rgb, LabColor)
        mixed_lab = convert_color(color_rgb, LabColor)
        # Calculate CIEDE2000 color difference
        distance = delta_e_cie2000(LabColor(target_lab.lab_l, target_lab.lab_a, target_lab.lab_b),
                                   LabColor(mixed_lab.lab_l, mixed_lab.lab_a, mixed_lab.lab_b))
        
        # should I return 1- distance to make it a loss function?
        # No, this distance is not out of 1
        return distance

    def update(self, color_volumes, observed_color, image):

        self.color_composition.append(color_volumes)
        self.images.append(image)
        self.observed_colors.append(sRGBColor(*observed_color, is_upscaled =True))
        
        color_score = self.color_score(observed_color)
        self.color_scores.append(color_score)
        
        self.optimizer.tell(np.array(self.color_composition),np.array( self.color_scores))
        self.optimal_proportions = [self.color_composition[i] for i in np.argmin(self.color_scores)]
       

    def propose_next_sample(self):
        return  self.optimizer.ask()

    def get_optimal_proportions(self):
         return self.optimal_proportions
    
    def visualize(self):

        #set up image display
        plt.ion()
        fig, ax = plt.subplots(1,2, figsize = (20,8))
        ax[0].set_title('Most Recent Image')
        ax[1].set_title('Color Loss Plot')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Loss')

        # get data in the right form for plotting
        norm_colors = [rgb.get_value_tuple() for rgb in self.observed_colors]
        loss_vals = self.color_scores
        
        image = self.images[-1]
        imgbuf = np.frombuffer(image, dtype = np.uint8)
        imgcv = cv2.imdecode(imgbuf, cv2.IMREAD_COLOR)
        imgcv_rgb = imgcv[:,:,[2,1,0]]

        # plot and update 
        for i, loss in enumerate(loss_vals):
            ax[1].scatter(i, loss_vals[i], marker = 'o', color = norm_colors[i], s = 200)
            ax[0].imshow(imgcv_rgb)

        fig.canvas.draw()
        fig.canvas.flush_events()

        return
    
    def run_campaign(self, number_of_iterations, robotic_platform, pipette, camera,
                     color_stocks, samples, starting_well = 0 ,save =True):

        assert len(samples.wells) > number_of_iterations, 'Error: Too many samples to test for number of wells in labware.'

        for i in range(number_of_iterations):
            well = samples[i+starting_well]
            # run point in real world
            print(f'Dispensing into well {well}')
            query_point = self.propose_next_sample()
            print(f'RYB values tested: {query_point}')

            observed_RGB, image = jubilee_protocols.sample_point(robotic_platform, pipette, camera, query_point,
                                                            self.sample_volume, well, color_stocks, save=save)

            print(f'RGB values observed: {observed_RGB}')
            self.update(self, query_point, observed_RGB, image)
            try:
                self.visualize()
            except Exception as e:
                print(e)
                pass
        
        return 
