"""
This file contains code that will kick off training and testing processes
"""
import os
import json
from sklearn.model_selection import train_test_split

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"/home/workspace/section1/out"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 32
        self.patch_size = 64
        self.test_results_dir = "/home/workspace/section2"

if __name__ == "__main__":
    # Get configuration

    # Create a config object
    c = Config()

    # Load data
    print("Loading data...")

    # LoadHippocampusData 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # Create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    
    # initial train/test split: 80/20 is a common ratio
    train_tmp, split['test'] = train_test_split(keys, test_size=0.2, random_state=0)
    # consecutive train split into train/validation: would use 20% for validation
    split['train'], split['val'] = train_test_split(train_tmp, test_size=0.20, random_state=0)

    # Set up and run experiment
    
    # Complete Class UNetExperiment
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    del data 

    # run training
    exp.run()

    # prep and run testing

    # Complete Test method
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

