import numpy as np


class Sampler(object):
    def __init__(self, x, chunk_size, skip_size, batch_size, shuffle_first=False):
        """ Responsible for providing data chunks to the <Trainer> class.

        A training iteration in a LSTM language model consists of processing a "chunk". This class supplies the
        <Trainer> with the chunks to process.

        Args:
            x(tensor): Shape (seq_len, 1). Sequence of word indices corresponding to corpus.
            chunk_size(int): The maximum size of a chunk.
            batch_size(int): The number of batches to divide <x> into.
            shuffle_first(bool): Whether to shuffle the list of chunks first before processing them.
        """

        self.data_list = self._create_chunk_list(x, chunk_size, skip_size, batch_size)
        self.data_copy = self.data_list
        self.batch_size = batch_size
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
        self.data_list = self.data_copy
        if self.shuffle_first:
            self.shuffle()

    def shuffle(self):
        perm = np.random.permutation(len(self.data_list))
        self.data_list = [self.data_list[i] for i in perm]

    def _create_chunk_list(self, x, chunk_size, batch_size, skip_size=1):
        """
        Args:
            x(tensor): Shape (seq_len, 1).
            chunk_size(int): This corresponds to the look-back period of truncated BPTT i.e. the K2 parameter.
            skip_size(int): The number of time steps to skip in truncated BPTT i.e. the K1 parameter.

        Returns:
            chunk_list(list): [chunk(tensor)]; chunk shape (chunk_size, batch_size) or (remainder, batch_size).
        """

        x = self._batchify(x, batch_size)  # shape (new_seq_len, batch_size)

        chunk_list = []
        create_chunk_is_possible = True
        boundary = [0, chunk_size]

        while create_chunk_is_possible:
            inputs = x[boundary[0]: boundary[1]]
            targets = x[boundary[0] + 1: boundary[1] + 1]

            if inputs.shape[0] > 1:
                if inputs.shape[0] < chunk_size:
                    inputs = input[:-1]  # make sure inputs has same size as targets.

                chunk = (inputs, targets)
                chunk_list.append(chunk)
                boundary = [boundary[0] + skip_size, boundary[1] + skip_size]
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
