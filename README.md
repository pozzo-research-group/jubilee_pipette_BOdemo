# Jubilee Pipette-based color matching demonstration

This rather poorly named repository implements a color matching autonomous experimentation demonstration on the [science-jubilee](https://machineagency.github.io/science-jubilee/) laboratory automation platform. 


Color matching is an autonomous experiment in which a system learns to create a target color by mixing stock solutions of base colors in appropriate ratios. Our implementation uses acrylic paint as the base color. The Jubilee robot platform uses a pipette tool to transfer and mix the samples, and a raspberry pi camera tool to characterize the resulting color. This information is then used by a Bayesian optimization optimizer to suggest the next experiment to the Jubilee platform. 

## Running the experiment

This documentation assumes that you already have a Jubilee motion platform provisioned with a calibrated lab automation deck, OT2 pipette tool, and webcamera camera tool. If this doesn't describe you, head over to the [science-jubilee docs](https://machineagency.github.io/science-jubilee/getting_started/index.html#getting-started) to get this sorted out.

In addition to your Jubilee, you will need the following items. You should be able to substitute similar equipment with similar functionality, for example using a peristaltic pump dispensing tool instead of an OT2 pipette tool.

1. An assembled and provisioned Jubilee motion platform with a calibrated lab automation deck plate
2. An OT2 Pipette tool and webcamera tool on your Jubilee, with tool offsets and tool-specific calibrations completed.
3. Acrylic paint in red, yellow, blue, white, and black. No need for anything fancy here
4. A 96 well plate (~400 uL well volumes)
5. A 12 scintillation vial holder labware, printable from the [Pozzo group labware repo](https://github.com/pozzo-research-group/Automation-Hardware/tree/master/Vial%20Holders/20mlscintillation_12_wellplate_18000ul).
6. Opentrons tiprack, with tips, for your pipette tool
7. 5 20cc scintillation vials for holding paint stocks

### 1. Install this library

Install this library by running `git clone https://github.com/pozzo-research-group/jubilee_pipette_BOdemo.git`, changing directory into jubilee_pipette_BOdemo, then doing a pip development install with `pip install -e .`. Also make sure that you have installed [science-jubilee](https://github.com/machineagency/science-jubilee) into the same environment. 

### 2. Install the bo-serving library
 This step is somewhat optional. By default, this experiment interfaces with a Bayesian Optimizer implemented as an HTTP endpoint. This in theory makes it easier to decouple the experiment execution hardware from the ML hardware and makes it possible to swap in new BO/other optimization implementations. In practice, the code for this experiment and the BO endpoint are still tightly coupled. You will need to use our [bo-serving](https://github.com/pozzo-research-group/bo-serving?tab=readme-ov-file) library if you choose to go this route. Follow the documentation at that repository to install the library and spin up an HTTP endpoint for Bayesian optimization. Alternatively, see below for using a 'local' optimizer.

 ### 3. Perform any necessary labware calibration

 You may need to perform additional labware calibration to get good alignment on Jubilee. Follow the [labware offset calibration instructions](https://machineagency.github.io/science-jubilee/getting_started/deck_guide.html#deck-guide) from the science-jubilee documentation. 


### 4. Launch the experiment notebook and follow it

The experiment is run from the `RobRoss_Happy_Little_Accidents.ipynb` notebook. Launch it and follow the steps to match your physical Jubilee setup to the python jubilee system setup. 

### 5. Run the Streamlit dashboard to view results

The live results of the experiment are viewable in a Streamlit dashboard. To run this, launch the `streamlit_dash.py` file from the streamlit_dash directory. This is done with the command `streamlit run streamlit_dash.py`




## Guide to code structure

The organization attempts to modularize the various components of the experimental workflow. Experimental orchestration is handled in `color_matcher.py` by the ColorMatcher class. This class has methods to handle experiment setup, dispatch experiments to the Jubilee driver, and update/query the Bayesian optimizer.

Interactions with the physical Jubilee platform are handled from the `Jubilee_protocols.py` file. 

Functions to process the captured images and return an RGB value are in the `image_processing.py` file.

`in_silico_mixing.py` has functions to perform simulated mixing, allowing you to run the color match experiment in a completely simulated environment without a Jubilee. This is great for debugging BO/optimization algorithms or other implementation details. To use in-silico mixing, pass the the keyword argument `in_silico_mixing = True` to the ColorMatcher constructor, along with the keyword argument `in_silico_colors = [(RGB tuple 1), (RGB tuple 2), ..., (RGB tuple total_stocks)]`. `total_stocks` is the number of stock primary color solutions the experiment uses.

As mentioned above, there are a few options for Bayesian optimization implementations. The default is to use an optimizer implemented as an HTTP endpoint. You also have the option of using a basic optimizer implemented in [Ax](https://ax.dev/) or custom BO implementation in BoTorch. Unfortunately, the choice of optimizer is currently hard-coded in the ColorMatcher constructor function. To change it, change the line `self.optimizer = ...` in the ColorMatcher constructor to your choice of `AxSolver` (implemented in `ax_solver.py`) or `BaysOptimizer` (implemented in `solver.py`). You will also need to change the arguments to the optimizer constructor as well. 

