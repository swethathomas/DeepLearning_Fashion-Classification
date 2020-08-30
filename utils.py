import numpy as np
import os
import os.path as osp
import argparse

#this code intializes important parameters needed for the project
Config ={}
# path containing datset
#Config['root_path'] = 'polyvore_outfits_hw\polyvore_outfits'
Config['root_path']='/home/ubuntu/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''
Config['test_file']='test_category_hw.txt'

Config['train_compatability']='pairwise_compatibility_train.txt'
Config['valid_compatability']='pairwise_compatibility_valid.txt'
Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 5
Config['batch_size'] = 64

Config['learning_rate'] = 0.0001
Config['num_workers'] = 1

