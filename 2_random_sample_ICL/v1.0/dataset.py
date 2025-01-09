import deepchem
import deepchem.molnet

tasks, datasets, transformers = deepchem.molnet.load_clintox(splitter='scaffold', reload=True,
                                                             data_dir='./data/clintox_data',
                                                             save_dir='./data/clintox_datasets')

train_dataset, valid_dataset, test_dataset = datasets
