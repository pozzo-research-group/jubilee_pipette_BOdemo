import numpy as np
import torch

torch.set_default_dtype(torch.double)

from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from bot.models.utils import initialize_model


class BaysOptimzer():
    def __init__(self, bounds, batch_size):
        self.model_name = "gp"
        self.model = None
        self.acq_func = None
        self.initial_bounds = bounds
        self.tensor_bounds = torch.tensor(self.initial_bounds).transpose(-1, -2)
        
        self.batch = batch_size
        self.design_space_dim = len(bounds)
        self.output_dim = 1
        self.model_args =  {"model":self.model_name,
                            "num_epochs" : 2500,
                            "learning_rate" : 1e-3
                            }
    @staticmethod
    def data_utils(data):

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        else:
            data = torch.from_numpy(np.ndarray(data))
        
        return data

    def update(self, x_data, y_data):

        x_data = self.data_utils(x_data)
        y_data = -1*self.data_utils(y_data)

        gp_model = initialize_model(self.model, self.model_args, self.design_space_dim,self.output_dim) 

        normalized_x = normalize(x_data, self.tensor_bounds)
        gp_model = gp_model.fit(normalized_x, y_data)
        
        self.model = gp_model
    
        acquisition = qExpectedImprovement(self.model, best_f = 0.0)

        self.acq_func = acquisition
        
        return 
    
    def ask(self):

        standard_bounds = torch.tensor([(float(1e-5), 1.0) for _ in range(self.design_space_dim)]).transpose(-1, -2)

        normalized_candidates, acqf_values = optimize_acqf(
            self.acquisition, 
            standard_bounds, 
            q=self.batch, 
            num_restarts=20, 
            raw_samples=1024, 
            return_best_only=True,
            sequential=False,
            options={"batch_limit": 1, "maxiter": 10, "with_grad":True}
            )
        # calculate acquisition values after rounding
        new_x = unnormalize(normalized_candidates.detach(), bounds=self.tensor_bounds) 

        return new_x

    
