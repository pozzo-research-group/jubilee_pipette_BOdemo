import numpy as np
import torch

torch.set_default_dtype(torch.double)

from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler 
from botorch.optim import optimize_acqf
from botorch.sampling.stochastic_samplers import StochasticSampler 
from botorch.utils.transforms import normalize, unnormalize

from bot.models.utils import initialize_model


class BaysOptimizer():
    def __init__(self, bounds, batch_size, task = 'maximize'):
        self.model_name = "gp"
        self.model = None
        self.acq_func = None
        self.initial_bounds = bounds
        self.tensor_bounds = torch.tensor(self.initial_bounds).transpose(-1, -2)
        self.task = task
        self.batch = batch_size
        self.design_space_dim = len(bounds)
        self.output_dim = 1
        self.model_args =  {"model":self.model_name,
                            "num_epochs" : 100,
                            "learning_rate" : 1e-3,
                            "verbose": 0}
    @staticmethod
    def data_utils(data):

        if isinstance(data, np.ndarray):
            data_ = torch.from_numpy(data)
        else:
            data = torch.from_numpy(np.array(data))
        
        return data_

    def update(self, x_data, y_data):

        x_data = self.data_utils(x_data)

        if self.task == 'maximize':
            y_data = self.data_utils(y_data)
            best= y_data.max()
        elif self.task == 'minimize':
            y_data = -1*self.data_utils(y_data)
            best = y_data.min()
        else:
            raise ValueError(f'Task must be either maximize or minimize, not {self.task}')

        gp_model = initialize_model(self.model_name, self.model_args, self.design_space_dim, self.output_dim) 

        normalized_x = normalize(x_data, self.tensor_bounds)
        gp_model = gp_model.fit(normalized_x, y_data)
        
        self.model = gp_model

        sampler = StochasticSampler(torch.Size([128]))
        acquisition = qExpectedImprovement(self.model, best_f = best, sampler= sampler)

        self.acq_func = acquisition
        
        return 
    
    def ask(self):

        indices = torch.arange(self.tensor_bounds.shape[1])
        coeffs = torch.ones(self.tensor_bounds.shape[1])
        constraints = [(indices, coeffs, 1)]

        normalized_candidates, acqf_values = optimize_acqf(
            self.acq_func, 
            self.tensor_bounds, 
            q=self.batch, 
            num_restarts=5, 
            raw_samples=10, 
            return_best_only=True,
            sequential=False,
            options={"batch_limit": 1, "maxiter": 10, "with_grad":True}, 
            equality_constraints=constraints
            )
        # calculate acquisition values after rounding
        new_x = unnormalize(normalized_candidates.detach(), bounds=self.tensor_bounds) 

        return new_x.numpy().squeeze()

    
