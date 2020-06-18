"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import sys
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"YOUR DIRECTORY HERE"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "RESULTS GO HERE"

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()
    c.root_dir = '/home/workspace/data'
    c.test_results_dir = '/home/workspace/out'
    c.n_epochs = 5

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # <YOUR CODE GOES HERE>
    
    # Attention! Split is performed with random shuffling (by randomly selecting indices)
    # DataLoader uses shuffling=True but I noticed it later, and this shuffling should not hurt anything
    keys_all = np.arange(len(data))
    
    np.random.seed(0)
    
    train_size = int(len(keys_all) * .6)
    train_keys = np.random.choice(keys_all, size=train_size, replace=False)    
    
    indices = np.zeros(keys_all.shape, dtype=bool)
    indices[train_keys] = True
    indices_non_train = ~indices
    non_train_keys = keys_all[indices_non_train]
    
    valid_size = int(len(non_train_keys) / 2)
    valid_keys = np.random.choice(non_train_keys, size=valid_size, replace=False)
    
    indices[valid_keys] = True
    indices_test = ~indices
    test_keys = keys_all[indices_test]

    print(f'Random split. Train: {len(train_keys)}, valid: {len(valid_keys)}, test: {len(test_keys)}')
    assert np.intersect1d(train_keys, valid_keys).tolist() == [], 'Intersection train & valid not empty'
    assert np.intersect1d(train_keys, test_keys).tolist() == [], 'Intersection train & test not empty'
    assert np.intersect1d(valid_keys, test_keys).tolist() == [], 'Intersection valid & test not empty'
    
    sorted_union = np.sort(np.union1d(np.union1d(train_keys, valid_keys), test_keys))
    assert sorted_union.tolist() == np.arange(len(data)).tolist(), 'Union does not contain all keys'

    split['train'] = train_keys
    split['val'] = valid_keys
    split['test'] = test_keys
    
    
    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    json_filename = os.path.join(exp.out_dir, "results.json")
    with open(json_filename, 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))
    print(f'Done. Results written to: {json_filename}')

