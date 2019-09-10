from collections import defaultdict
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import math
import os
import torch
import sys

"""
additional features:
- rescaling the learning rate.
- sampler iterates over variable sized chunks.
"""


class Sampler(object):
    def __init__(self, x, avg_chunk_size, skip_size, batch_size, shuffle_first=False):
        """ Responsible for providing data chunks to the <Trainer> class.

        A training iteration in a LSTM language model consists of processing a "chunk". This class supplies the
        <Trainer> with the chunks to process.

        Args:
            x(tensor): Shape (seq_len, 1). Sequence of word indices corresponding to corpus.
            chunk_size(int): The maximum size of a chunk.
            batch_size(int): The number of batches to divide <x> into.
            shuffle_first(bool): Whether to shuffle the list of chunks first before processing them.
        """
        self.x = x
        self.avg_chunk_size = avg_chunk_size
        self.skip_size = skip_size
        self.batch_size = batch_size
        self.data_list = self._create_chunk_list()
        self.data_copy = self.data_list
        self.shuffle_first = shuffle_first
        self.reset()

    def __iter__(self):  # implement iterator interface.
        return self

    def __next__(self):
        return self.sample()

    def __len__(self):
        actual = len(self.data_copy) / self.batch_size
        rounded = len(self.data_copy) // self.batch_size
        if actual - rounded == 0:
            return rounded
        return rounded + 1

    def sample(self):
        if self.has_next():
            batch = self.data_list[:self.batch_size] # return top
            self.data_list = self.data_list[self.batch_size:] # remove top
            return batch
        else:
            raise StopIteration()

    def has_next(self):
        if len(self.data_list) > 0:
            return True
        self.reset()  # fill up again for next time you want to iterate over it.
        return False

    def reset(self):
        self.data_list = self._create_chunk_list()
        self.data_copy = self.data_list
        if self.shuffle_first:
            self.shuffle()

    def shuffle(self):
        perm = np.random.permutation(len(self.data_list))
        self.data_list = [self.data_list[i] for i in perm]

    def _create_chunk_list(self):
        """
        Different from other <lstm.Sampler> this sampler creates variable length chunks.

        Args:
            x(tensor): Shape (seq_len, 1).
            chunk_size(int): This corresponds to the look-back period of truncated BPTT i.e. the K2 parameter.
            skip_size(int): The number of time steps to skip in truncated BPTT i.e. the K1 parameter.

        Returns:
            chunk_list(list): [chunk(tensor)]; chunk shape (chunk_size, batch_size) or (remainder, batch_size).
        """

        x = self._batchify(self.x, self.batch_size)  # shape (new_seq_len, batch_size)

        chunk_list = []
        create_chunk_is_possible = True

        chunk_size = max(5, int(np.random.normal(self.avg_chunk_size, 5)))
        time = 0

        while create_chunk_is_possible:
            inputs = x[time: time + chunk_size]
            targets = x[time + 1: time + chunk_size + 1].reshape(-1)

            if inputs.shape[0] > 1:
                if inputs.shape[0] < chunk_size:
                    inputs = input[:-1]  # make sure inputs has same size as targets.
                chunk = (inputs, targets)
                chunk_list.append(chunk)

                time += self.skip_size
                chunk_size = max(5, int(np.random.normal(self.avg_chunk_size, 5)))

            else:
                create_chunk_is_possible = False

        return chunk_list

    @staticmethod
    def _batchify(x, batch_size):
        """
        Args:
            x(tensor): shape (old_seq_len, 1).

        Returns:
            x(tensor): shape (seq_len, batch_size).
        """
        seq_len = x.shape[0] // batch_size
        x = x.narrow(0, 0, batch_size * seq_len)  # (seq_len * batch_size, 1).
        x = x.reshape(seq_len, batch_size)  # (seq_len, batch_size)
        return x


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
        # todo: starting each bactch you have to detach.

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
                lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = lr * chunk[0].shape[0] / self.sampler.avg_chunk_size
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


class TrainerAWD(object):
    def __init__(self, data, avg_chunk_size, handler, device):
        self.device = device
        self.handler = handler
        self.data = data
        self.avg_chunk_size = avg_chunk_size

    def _train_iter(self, chunk, hidden_list, model, optimizer):
        # todo: starting each bactch you have to detach.

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

    def get_batch(self, data, i, seq_len=None):
        seq_len = min(seq_len if seq_len else self.avg_chunk_size, len(data) - 1 - i)
        inputs = data[i:i + seq_len]
        targets = data[i + 1:i + 1 + seq_len].view(-1)
        return inputs, targets

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors,
        to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return [self.repackage_hidden(v) for v in h]

    def __call__(self, model, optimizer, current_epoch):

        hidden_list = model.init_hidden_list()

        with tqdm(total=self.data.shape[0] - 1 - 1) as pbar_train:
            for i in enumerate(self.data.shape[0] - 1 - 1):
                chunk_size = self.avg_chunk_size if np.random.random() < 0.95 else self.avg_chunk_size / 2.
                chunk_size = max(5, int(np.random.normal(chunk_size, 5)))
                lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = lr * chunk_size / self.avg_chunk_size  # rescale learning rate.
                model.train()
                inputs, targets = self.get_batch(self.data, i, seq_len=chunk_size)
                hidden_list = self.repackage_hidden(hidden_list)
                optimizer.zero_grad()

                # now some model specific stuff:

        #### OLD:
        batch_stats = defaultdict(lambda: [])
        epoch_stats = OrderedDict({})
        hidden_list = model.init_hidden_list(self.sampler.batch_size)

        with tqdm(total=len(self.sampler)) as pbar_train:
            for i, chunk in enumerate(self.sampler):
                lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = lr * chunk[0].shape[0] / self.sampler.avg_chunk_size
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


class ExperimentIO(object):

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
        state['init_params'] = model.__dict__
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
