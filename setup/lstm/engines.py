import os
import torch
from torch import optim
import torch.nn as nn

"""
Description:

- Cross Entropy Loss.
- Multi-Layer LSTM.
- Adam optimizer.
- Fixed length TBPTT.
- No regularization.
"""


class DefaultEngine(object):
    def __init__(self, config):
        trial_parent_dir = config['trial_parent_dir']
        trial = config['trial']
        self.checkpoints_parent_dir = os.path.join(trial_parent_dir, 'trial_{}'.format(trial))
        self.training_experiment = None

        if not os.path.exists(self.checkpoints_parent_dir):
            os.makedirs(self.checkpoints_parent_dir)

        config_filename = os.path.join(self.checkpoints_parent_dir, 'config.txt')
        with open(config_filename, 'w+') as f:
            for k, v in config.items():
                f.write('{}: {}\n'.format(k, v))

    def add_training_experiment(self, training_experiment):
        self.training_experiment = training_experiment

    def run(self):
        experiment_results_filename = os.path.join(self.training_experiment.experiment_dirname, 'results.txt')
        if not os.path.exists(experiment_results_filename) or self.training_experiment.resume:
            self.training_experiment.run()
        else:
            print('{} exists. delete folder to re-run or resume.'.format(experiment_results_filename))


class Engine(DefaultEngine):

    def __init__(self, config):
        super(Engine, self).__init__(config)

        from setup.lstm.models import LSTM
        from setup.lstm.trainer import Experiment
        import dataset
        from setup.lstm.trainer import Sampler
        from setup.lstm.trainer import Trainer
        from setup.lstm.trainer import Tester

        data = dataset.BUILTIN_DATASET[config['dataset_name']]
        train_data = data.get('train')  # tensor shape (seq_len, 1)
        valid_data = data.get('valid')

        train_sampler = Sampler(train_data, config['chunk_size'], config['skip_size'], config['batch_size'])
        valid_sampler = Sampler(valid_data, config['chunk_size'], config['skip_size'], config['batch_size'])
        trainer = Trainer(train_sampler, data.vocab_size, device=torch.device(0))
        tester = Tester(valid_sampler, data.vocab_size, device=torch.device(0))

        model = LSTM(
            input_size=config['model']['input_size'],  # size of embeddings.
            hidden_size=config['model']['hidden_size'],
            vocab_size=data.vocab_size,
            n_layers=config['model']['n_layers'],
            device=torch.device(0))

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        training_experiment = Experiment(
            model=model,
            optimizer=optimizer,
            num_epochs=config['num_epochs'],
            trainer_module=trainer,
            tester_module=tester,
            experiment_dirname=self.checkpoints_parent_dir,
            use_gpu=True)

        self.add_training_experiment(training_experiment)