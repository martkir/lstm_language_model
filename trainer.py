from collections import defaultdict
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import math
import os
import torch
import sys


class Handler(object):
    def __init__(self, criterion, vocab_size):
        self.criterion = criterion
        self.vocab_size = vocab_size

    def compute_loss(self, scores, targets):
        loss = self.criterion(scores.reshape(-1, self.vocab_size), targets)
        return loss

    @staticmethod
    def process_chunk(chunk, model):
        inputs, targets = chunk
        inputs = inputs.to(device=model.device)
        targets = inputs.to(device=model.device)
        return inputs, targets


class Trainer(object):
    def __init__(self, sampler, handler, device):
        self.sampler = sampler
        self.device = device
        self.handler = handler

    def _train_iter(self, chunk, hidden_list, model, optimizer):
        model.train()
        inputs, targets = self.handler.process_chunk(chunk, model)
        scores, hidden_list = model.forward(inputs, hidden_list)
        loss = self.handler.compute_loss(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats_dict = {
            'train_loss': loss.data.cpu().numpy(),
            'train_ppl': math.exp(loss.data.cpu().numpy())  # perplexity.
        }

        return stats_dict, hidden_list, model, optimizer

    def __call__(self, model, optimizer, current_epoch):
        batch_stats = defaultdict(lambda: [])
        epoch_stats = OrderedDict({})
        hidden_list = model.init_hidden_list(self.sampler.batch_size)

        with tqdm(total=len(self.sampler)) as pbar_train:
            for i, chunk in enumerate(self.sampler):
                output, hidden_list, model, optimizer = self._train_iter(chunk, hidden_list, model, optimizer)

                for k, v in output.items():
                    batch_stats[k].append(output[k])

                description = 'epoch: {}'.format(current_epoch)
                description += ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_stats.items()])
                description += 'lr: {}'.format(optimizer.params_group[0]['lr'])

                pbar_train.update(1)
                pbar_train.set_description(description)

        for k, v in batch_stats.items():
            epoch_stats[k] = np.around(np.mean(v), decimals=4)

        return model, optimizer, epoch_stats


class Tester(object):

    def __init__(self, sampler, handler, device):
        self.sampler = sampler
        self.handler = handler
        self.device = device

    def _valid_iter(self, chunk, hidden_list, model):
        model.eval()
        inputs, targets = self.handler.process_chunk(chunk, model)
        scores, hidden_list = model.forward(inputs, hidden_list)
        loss = self.handler.compute_loss(scores, targets)

        stats_dict = {
            'train_loss': loss.data.cpu().numpy(),
            'train_ppl': math.exp(loss.data.cpu().numpy())  # perplexity.
        }

        return stats_dict, hidden_list

    def __call__(self, model, current_epoch):
        batch_stats = defaultdict(lambda: [])
        results = OrderedDict({})
        hidden_list = self.handler.init_hidden_list(self.sampler.batch_size)

        with tqdm(total=len(self.sampler)) as pbar:
            for i, chunk in enumerate(self.sampler):
                stats_dict, hidden_list = self._valid_iter(model, chunk, hidden_list)
                for k, v in stats_dict.items():
                    batch_stats[k].append(stats_dict[k])

                description = 'epoch: {}'.format(current_epoch)
                description += ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_stats.items()])

                pbar.update(1)
                pbar.set_description(description)

        for k, v in batch_stats.items():
            results[k] = np.around(np.mean(v), decimals=4)

        return results


class ExperimentIO(object):  # todo: fix the loading method.

    def __init__(self):
        pass

    @staticmethod
    def load_model(model_class, filename):
        state = torch.load(f=filename)
        init_params = state['init_params']
        model = model_class(**init_params)
        model.load_state_dict(state_dict=state['network'])
        return model

    @staticmethod
    def load_checkpoint(model, optimizer, filename):
        state = torch.load(f=filename)
        model.load_state_dict(state_dict=state['network'])
        optimizer.load_state_dict(state_dict=state['optimizer'])
        return model, optimizer

    @staticmethod
    def save_checkpoint(model, optimizer, current_epoch, dirname):
        state = dict()
        state['network'] = model.state_dict()  # save network parameter and other variables.
        state['init_params'] = model.init_params
        state['optimizer'] = optimizer.state_dict()

        filename = os.path.join(dirname, 'epoch_{}'.format(current_epoch))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(state, f=filename)

    @staticmethod
    def save_epoch_stats(epoch_stats, filename, first_line=False):
        """
        tasks:
        - if directory does not exist it will be created.
        - if file already exists then content get's
        params:
            epoch_stats: dict {}
        remarks:
            (1) use mode = +w to overwrite.
            (2) such that column names are in a desired order.
        """

        if type(epoch_stats) is not OrderedDict:  # (2)
            raise Exception('epoch_stats must be an ordered dict. got: {}'.format(type(epoch_stats)))

        if first_line:
            mode = '+w'
        else:
            mode = '+a'

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(filename, mode) as f:
            if first_line:
                header = ','.join([k for k in epoch_stats.keys()])
                f.write(header + '\n')

            line = ','.join(['{:.4f}'.format(value) for value in epoch_stats.values()])
            f.write(line + '\n')


class Experiment(object):  # todo: to re-implement.

    def __init__(self,
                 model,
                 optimizer,
                 num_epochs,
                 trainer_module,
                 tester_module,
                 experiment_dirname,
                 use_gpu=True,
                 resume=False):

        self.model = model
        self.optimizer = optimizer
        self.experiment_dirname = experiment_dirname
        self.num_epochs = num_epochs
        self.results_filename = os.path.join(experiment_dirname, 'results.txt')
        self.model_dirname = os.path.join(experiment_dirname, 'checkpoints')
        self.device = torch.device('cpu')  # default device is cpu.
        self.resume = resume

        self.trainer_module = trainer_module
        self.tester_module = tester_module

        device_name = 'cpu'
        if use_gpu:
            if not torch.cuda.is_available():
                print("GPU IS NOT AVAILABLE")
            else:
                self.device = torch.device('cuda:{}'.format(0))
                device_name = torch.cuda.get_device_name(self.device)
                self.model.to(device=self.device)  # (1)

        print('initialized trainer with device: {}'.format(device_name))

    def run(self):
        next_epoch = 0
        if self.resume:
            num_lines = sum(1 for _ in open(self.results_filename))
            if num_lines > 0:
                start_epoch = num_lines - 2
                model_filename = os.path.join(self.model_dirname, 'epoch_{}'.format(start_epoch))
                self.model, self.optimizer = ExperimentIO.load_checkpoint(self.model, self.optimizer, model_filename)
                self.model.to(device=self.device)
                next_epoch = start_epoch + 1

        end_epoch = next_epoch + self.num_epochs
        for current_epoch in range(next_epoch, end_epoch):
            # valid_results = self.tester_module(self.model, current_epoch)
            self.model, self.optimizer, train_results = self.trainer_module(self.model, self.optimizer, current_epoch)
            if current_epoch % 1 == 0 and current_epoch > -1:
                valid_results = self.tester_module(self.model, current_epoch)
            else:
                valid_results = {}

            sys.stderr.write('\n')
            results = OrderedDict({})
            results['current_epoch'] = current_epoch
            for k in train_results.keys():
                results[k] = train_results[k]
            for k in valid_results.keys():
                results[k] = valid_results[k]

            ExperimentIO.save_checkpoint(self.model, self.optimizer, current_epoch, dirname=self.model_dirname)
            ExperimentIO.save_epoch_stats(results, self.results_filename, first_line=(current_epoch == 0))
