import setup.lstm.engines


def train_vanilla_lstm():

    config = {
        'dataset_name': 'wikitext-2',
        'chunk_size': 35,
        'skip_size': 35,
        'batch_size': 20,
        'input_size': 64,  # size of word embeddings.
        'hidden_size': 32,
        'n_layers': 1,
        'lr': 0.001,  # learning rate
        'num_epochs': 2,
        'experiment_dirname': 'results/vanilla_lstm',  # where checkpoints and training/ validation results are saved.
    }

    engine = setup.lstm.engines.Engine(config)
    engine.run()

