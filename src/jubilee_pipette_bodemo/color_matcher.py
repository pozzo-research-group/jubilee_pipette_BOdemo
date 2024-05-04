import cv2
import json
import jubilee_pipette_bodemo.jubilee_protocols as jubilee_protocols

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from datetime import date
#from jubilee_pipette_bodemo.solver import BaysOptimizer
from jubilee_pipette_bodemo.ax_solver import AxSolver 
from jubilee_pipette_bodemo.http_solver import HTTPSolver
from jubilee_pipette_bodemo.solver import BaysOptimizer
from jubilee_pipette_bodemo import in_silico_mixing




# Workaround: https://github.com/gtaylor/python-colormath/issues/104
def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)


class ColorMatcher:
    def __init__(self, total_stocks, sample_volume, score_type = 'euclidean', task = 'minimize', n_random_its = 5, n_bo_its = 20, in_silico_mixing = False, in_silico_colors = None, http_url = 'http://localhost:5000'):
        self.nstocks = total_stocks
        self.target_color = None
        self.sample_volume = sample_volume
        self.optimal_proportions = None
        self.score_type = score_type
        self.task = task
        self.observed_colors = []
        self.sample_composition = []
        self.color_scores = []
        self.images = []
        # Initialize optimizer
        self.optimizer = BaysOptimizer([(0,1.0)] * self.nstocks, 1, task = 'minimize')
        self.model = None
        self.in_silico_mixing = in_silico_mixing
        self.in_silico_colors = in_silico_colors

        if in_silico_mixing:
            assert len(in_silico_colors) == total_stocks, "When using in silico mixing, you must supply a list of RGB colors `in_silico_colors` with the same number of colors as `total_stocks`"

    def select_target_color(self):
        
        def pick_a_color():
            color_picker = widgets.ColorPicker(
                concise=False,
                description='Pick a color',
                value='blue',
                disabled=False
            )
            return color_picker

        color_picker = pick_a_color()
        target_rgb_output = []

        def on_color_change(change):
            hexcode = change['new']
            target_rgb = tuple(int(hexcode.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            target_rgb_output.clear()
            target_rgb_output.extend(target_rgb)
        color_picker.observe(on_color_change, names='value')
        
        display(color_picker)
    
        self.target_color = target_rgb_output
        return target_rgb_output
    
    def generate_initial_data(self, n_samples):
        # Generate n_samples random color samples presented as proportions of stock colors volumes
        color_samples = np.random.dirichlet(np.ones(self.nstocks), n_samples)[0]
        return color_samples
        #return self.optimizer.ask()

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

        ## Update the optimizer with the initial data
        self.optimizer.update(np.asarray(self.sample_composition), np.asarray(self.color_scores).reshape(-1,1))

        return 

    def color_score(self, color, score_type = 'euclidean'):
        # color : list of rgb values of sampled color
        
        target_rgb = sRGBColor(*self.target_color, is_upscaled =True if max(self.target_color) > 1 else False)
        color_rgb = sRGBColor(*color, is_upscaled =True if max(color) > 1 else False)
        if score_type == 'CIE2000':
            # Convert RGB color to Lab color (CIE L*a*b* color space)
            target_lab = convert_color(target_rgb, LabColor)
            mixed_lab = convert_color(color_rgb, LabColor)
            # Calculate CIEDE2000 color difference
            distance = delta_e_cie2000(LabColor(target_lab.lab_l, target_lab.lab_a, target_lab.lab_b),
                                    LabColor(mixed_lab.lab_l, mixed_lab.lab_a, mixed_lab.lab_b))
            return distance

        elif score_type == 'euclidean':  

            distance = [np.abs(np.array(t) - np.array(m)) for t, m in zip(target_rgb.get_value_tuple(), color_rgb.get_value_tuple())]
            score = np.linalg.norm(distance)

            return score

    def update(self, color_volumes, observed_color, image = None):

        self.sample_composition.append(list(color_volumes))
        self.observed_colors.append(observed_color)
        
        color_score = self.color_score(observed_color, score_type = self.score_type)
        self.color_scores.append(color_score)
        self.images.append(image)

               
    def propose_next_sample(self):

        next_sample = self.optimizer.ask()

        return next_sample

    def get_optimal_proportions(self):
        self.optimal_proportions = [self.sample_composition[i] for i in np.argmin(self.color_scores)]
        return self.optimal_proportions
    
    def visualize(self, fig, ax):

        # get data in the right form for plotting

        norm_colors = [[c/255 for c in rgb] for rgb in self.observed_colors]
        loss_vals = self.color_scores
        
        image = self.images[-1]
        imgbuf = np.frombuffer(image, dtype = np.uint8)
        imgcv = cv2.imdecode(imgbuf, cv2.IMREAD_COLOR)
        imgcv_rgb = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)

        # plot and update 
        ax[1].scatter(range(len(loss_vals)), loss_vals, marker = 'o', color = norm_colors, s = 200)
        ax[0].imshow(imgcv_rgb)

        fig.canvas.draw()
        fig.canvas.flush_events()

        return
    
    def run_campaign(self, number_of_iterations, robotic_platform, pipette, camera,
                     color_stocks, samples, starting_well = 0 ,save =True, saveToFile = True):

        if not self.in_silico_mixing:
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
            if not self.in_silico_mixing:
                well = samples[i+starting_well]
            # run point in real world
            if not self.in_silico_mixing:
                print(f'Dispensing into well {well}')

            if len(self.sample_composition) < 10:
                query_point = self.generate_initial_data(1)
                print(query_point)
            else:
                query_point = self.propose_next_sample()

            print('query point: ', query_point)
            print('type query pt: ', type(query_point))

            if sum(query_point) == 0:
                print('All stock volumes are zero. Skipping this iteration.')
                self.optimizer.tell(query_point, 10e5) # will need to check that this is ok!
            else:
                print(f'RYB values tested: {query_point}')

                if not self.in_silico_mixing:
                    observed_RGB, image = jubilee_protocols.sample_point(robotic_platform, pipette, camera, query_point,
                                                                    self.sample_volume, well, color_stocks, save=save)
                else:
                    observed_RGB, image = in_silico_mixing.sample_point(query_point, self.in_silico_colors)

                print(f'RGB values observed: {observed_RGB}')
                self.update(query_point, observed_RGB, image)
                ## Update the optimizer with data
                print('color score: ', self.color_scores)
                print('type: ', type(self.color_scores))
                self.optimizer.update(np.array(self.sample_composition), np.array(self.color_scores).reshape(-1,1))
                
                try:
                    self.visualize(fig, ax)
                except Exception as e:
                    print(e)
                    pass
                    
                if not self.in_silico_mixing:
                    data['Sample_id'] = f'{well.name}_{well.slot}'
                data['Stock_volumes'] = list(query_point)
                data['RGB_measured'] = observed_RGB
                data['Score'] = self.color_scores[-1]
                data_to_save.append(data)

            if saveToFile ==True:
                td = date.today().strftime("%Y%m%d")
                filename = f'{td}_ColorMatcher_results.jsonl'
                with open(filename, 'wt') as f:
                    for entry in data_to_save:
                        f.write(json.dumps(entry) + '\n')      
        
        return 