import cv2
import json
import jubilee_protocols
import matplotlib.pyplot as plt
import numpy as np

from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from datetime import date
from solver import BaysOptimizer


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
        self.sample_composition = []
        self.color_scores = []
        self.images = []
        # Initialize optimizer
        self.optimizer = BaysOptimizer([(0,1)] * self.nstocks, bathc_size = 1) 
        self.model = self.optimizer.model

    def generate_initial_data(self, n_samples):
        # Generate n_samples random color samples presented as proportions of stock colors volumes
        color_samples = np.random.dirichlet(np.ones(self.nstocks), n_samples)
        return color_samples

    def run_initial_data(self, robotic_platform, pipette, camera, initial_data,
                     color_stocks, sample_wells, starting_well = 0 , save =True, saveToFile = True):

        assert len(sample_wells.wells) > len(initial_data), 'Error: Too many samples to test for number of wells in labware.'

        self.initial_data= []

        for i in range(len(initial_data)):
            data={}
            well = sample_wells[i+starting_well]
            # run point in real world
            print(f'Dispensing into well {well}')
            query_point = initial_data[i]
            observed_RGB, image = jubilee_protocols.sample_point(robotic_platform, pipette, camera, query_point,
                                                            self.sample_volume, well, color_stocks, save=save)

            print(f'RGB values observed: {observed_RGB}')
            self.update(query_point, observed_RGB, image)
    
            data['Sample_id'] = f'{well.name}_{well.slot}'
            data['Stock_volumes'] = list(query_point)
            data['RGB_measured'] = observed_RGB
            self.initial_data.append(data)        

        if saveToFile ==True:
            td = date.today().strftime("%Y%m%d")
            filename = f'{td}_initial_data_random_{len(initial_data)}.jsonl'
            with open(filename, 'wt') as f:
                 for entry in self.initial_data:
                    f.write(json.dumps(entry) + '\n')

        return 

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
        
        return distance

    def update(self, color_volumes, observed_color, image):

        self.sample_composition.append(list(color_volumes))
        self.images.append(image)
        self.observed_colors.append(sRGBColor(*observed_color, is_upscaled =True))
        
        color_score = self.color_score(observed_color)
        self.color_scores.append(color_score)

        ## Update the optimizer with the initial data
        self.optimizer.update(np.array(self.color_composition), np.array( self.color_scores))
        self.optimal_proportions = [self.color_composition[i] for i in np.argmin(self.color_scores)]
               
    def propose_next_sample(self):

        next_sample = self.optimizer.ask()
        if sum(next_sample) == 0:
            normed_sample = [0,0,0]
        else:
            normed_sample = [x/sum(next_sample) for x in next_sample]   

        return  normed_sample

    def get_optimal_proportions(self):
         return self.optimal_proportions
    
    def visualize(self, fig, ax):

        # get data in the right form for plotting
        norm_colors = [rgb.get_value_tuple() for rgb in self.observed_colors]
        loss_vals = self.color_scores
        
        image = self.images[-1]
        imgbuf = np.frombuffer(image, dtype = np.uint8)
        imgcv = cv2.imdecode(imgbuf, cv2.IMREAD_COLOR)
        imgcv_rgb = imgcv[:,:,[2,1,0]]

        # plot and update 
        ax[1].scatter(range(len(loss_vals)), loss_vals, marker = 'o', color = norm_colors, s = 200)
        ax[0].imshow(imgcv_rgb)

        fig.canvas.draw()
        fig.canvas.flush_events()

        return
    
    def run_campaign(self, number_of_iterations, robotic_platform, pipette, camera,
                     color_stocks, samples, starting_well = 0 ,save =True, saveToFile = True):

        assert len(samples.wells) > number_of_iterations, 'Error: Too many samples to test for number of wells in labware.'

        data_to_save = []
        data_to_save.append("------ Target Color ------")
        data_to_save.append("RGB: " + str(self.target_color))
        data_to_save.append("Sample Volume: " + str(self.sample_volume) + "ul")
        data_to_save.append("------ Campaign Data ------")
        
        #set up image display
        plt.ion()
        fig, ax = plt.subplots(1,2, figsize = (20,8))
        ax[0].set_title('Most Recent Image')
        ax[1].set_title('Color Loss Plot')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Loss')
               
        for i in range(number_of_iterations):
            data = {}
            well = samples[i+starting_well]
            # run point in real world
            print(f'Dispensing into well {well}')
            query_point = self.propose_next_sample()

            if sum(query_point) == 0:
                print('All stock volumes are zero. Skipping this iteration.')
                self.optimizer.tell(query_point, 10e5) # will need to check that this is ok!
            else:
                print(f'RYB values tested: {query_point}')

                observed_RGB, image = jubilee_protocols.sample_point(robotic_platform, pipette, camera, query_point,
                                                                self.sample_volume, well, color_stocks, save=save)

                print(f'RGB values observed: {observed_RGB}')
                self.update(query_point, observed_RGB, image)
                
                try:
                    self.visualize(fig, ax)
                except Exception as e:
                    print(e)
                    pass
                    
                data['Sample_id'] = f'{well.name}_{well.slot}'
                data['Stock_volumes'] = list(query_point)
                data['RGB_measured'] = observed_RGB
                data_to_save.append(data)

            if saveToFile ==True:
                td = date.today().strftime("%Y%m%d")
                filename = f'{td}_ColorMatcher_results.jsonl'
                with open(filename, 'wt') as f:
                    for entry in data_to_save:
                        f.write(json.dumps(entry) + '\n')      
        
        return 