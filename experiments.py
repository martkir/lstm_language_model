import setup.lstm.engines

"""
experiment ideas:
- SGD vs. Adam. (which is better).
"""


def train_vanilla_lstm():
    config = {
        'dataset_name': 'wikitext-2',
        'batch_size': 20,
        'chunk_size': 35,
        'skip_size': 35,
        'learning_rate': 0.001,
        'num_epochs': 2,
        'model': {
            'input_size': 64,  # size of embeddings.
            'hidden_size': 32,
            'n_layers': 1
        },
    }

    engine = setup.lstm.engines.Engine(config)
    engine.run()

