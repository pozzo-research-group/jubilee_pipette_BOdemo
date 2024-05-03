from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.modelbridge.registry import ModelRegistryBase, Models

from ax.utils.testing.core_stubs import get_branin_experiment, get_branin_search_space

from ax.modelbridge.factory import get_GPEI

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render



class AxSolver():
   
    def get_ax_object(self, n_params, n_random_trials, n_bo_trials):

        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=n_random_trials,  # How many trials should be produced from this generation step
                    min_trials_observed=1,  # How many trials need to be completed to move to next model
                    max_parallelism=1,  # Max parallelism for this step
                    model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
                    model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
                ),
                # 2. Bayesian optimization step
                GenerationStep(
                    model = get_GPEI,
                    num_trials=n_bo_trials,  # No limitation on how many trials should be produced from this step
                    max_parallelism=1
                ),
            ]
        )
        ax_client = AxClient(generation_strategy = gs)

        experiment = ax_client.create_experiment(
                name="color_matching",
                parameters=[{"name":f"x{i+1}", "type":"range", "bounds":[0.0, 1.0], "value_type":"float"} for i in range(n_params-1)], 
                objectives={"euclidean": ObjectiveProperties(minimize=True)},
                parameter_constraints=[' + '.join([f'x{i+1}' for i in range(n_params-1)]) + " <= 1.0"]  # Optional.
        )

        return ax_client

    def __init__(self, n_params, n_random_trials, n_bo_trials):

        self.ax_client = self.get_ax_object(n_params, n_random_trials, n_bo_trials)


    def ask(self):
        query_point, trial_ind = self.ax_client.get_next_trial()
        self.open_trial = trial_ind
        ax_ratios = list(query_point.values())
        # 3rd volume is fixed by selection of other 2. But then BO isn't learning this parameter explicitly...
        r3 = 1 - sum(ax_ratios)
        ax_ratios.append(r3)
        return ax_ratios
    
    def update(self, x_data, y_data):

        y_dict = {'euclidean':(y_data[-1], 0)}

        self.ax_client.complete_trial(trial_index = self.open_trial, raw_data = y_dict)

        return




