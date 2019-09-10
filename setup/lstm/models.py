import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, n_layers, device=torch.device(0)):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.device = device
        self.lstms = []
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                self.lstms.append(torch.nn.LSTM(input_size, hidden_size, batch_first=False))
            else:
                self.lstms.append(torch.nn.LSTM(hidden_size, hidden_size, batch_first=False))

        self.lstms = torch.nn.ModuleList(self.lstms)
        self.scores_layer = nn.Linear(hidden_size, vocab_size)
        self.embedding_layer = nn.Embedding(vocab_size, input_size)
        self._init_weights()

    def forward(self, inputs, hidden_list):
        """
        Args:
            inputs: shape (seq_len, batch_size, x_dim). corresponds to words.
        """
        embedded_inputs = self.embedding_layer(inputs)
        outputs = embedded_inputs
        new_hidden_list = []
        for layer_idx, lstm in enumerate(self.rnns):
            outputs, hidden = lstm(outputs, hidden_list[layer_idx])
            new_hidden_list.append(hidden)
        scores = self.scores_layer(outputs.reshape(outputs.shape[0] * outputs.shape[1], outputs.shape[2]))
        scores = scores.reshape(scores.shape[0], scores.shape[1], scores.shape[2])
        return scores, new_hidden_list

    def init_hidden_list(self, batch_size):
        """
        Note: This function is used by the <Trainer>. A training iteration depends on the hidden states. In order to
        start training, the initial hidden states (default 0) have to be provided.
        """
        weight = next(self.parameters()).data
        hidden_list = []
        for layer_idx in range(self.n_layers):
            h_0 = weight.new(1, batch_size, self.hidden_size).zero_()  # same data type as weight.
            c_0 = weight.new(1, batch_size, self.hidden_size).zero_()
            hidden = (h_0, c_0)
            hidden_list.append(hidden)
        return hidden_list

    def _init_weights(self):
        init_range = 0.1
        self.embedding_layer.weight.data.uniform_(-init_range, init_range)
        self.scores_layer.bias.data.zero_()
        self.scores_layer.weight.data.uniform_(-init_range, init_range)



