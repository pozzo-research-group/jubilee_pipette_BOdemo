import requests



class HTTPOptimizer():
    """
    Interfaces with a remote bayesian optimization service to ask and update
    """

    def __init__(self, n_params, n_random_trials, n_bo_trials, url):
        self.n_params = n_params
        self.n_random_trials = n_random_trials
        self.n_bo_trials = n_bo_trials
        self.url = url
        self.uuid = None
        self.open_trial_index = None
        self.metric = 'euclidean'

        init_response = requests.post(url + '/new_experiment', json = {'n_params':self.n_params, 'n_random_trials':self.n_random_trials, 'n_bo_trials':self.n_bo_trials})

        assert init_response.status_code == 200, f'Error when creating new experiment on service. Error: {init_response.content}'
        self.uuid = init_response.json()['uuid']


    def ask(self):

        print('http optimizer is asking service')
        response = requests.post(self.url+'/get_next_trial', json = {'uuid':self.uuid}, timeout=60)

        assert response.status_code == 200, f'Error when getting next trial, {response.content}'

        next_experiment = response.json()

        self.open_trial_index = next_experiment['trial_index']
        print('new trial index: ', self.open_trial_index)
        params = next_experiment['parameterization']

        return list(params.values())





    def update(self, x_data, y_data):


        print('recieved y: ', y_data)

        data_package = {}
        data_package['uuid'] = self.uuid
        data_package['trial_index'] = self.open_trial_index
        data_package['metric'] = self.metric
        data_package['mean'] = float(y_data[-1])
        data_package['std'] = 0

        print('update data: ', data_package)
        response = requests.post(self.url + '/complete_trial', json = data_package, timeout = 10)

        assert response.status_code == 200, f'Error when updating trial, {response.content}'

        return




