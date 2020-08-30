from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

import tensorflow
from utils import Config

#class to read and preprocess image datafiles 
class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        self.le=LabelEncoder()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()


    #function to perform preprocessing on input image data files 
    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
    'test_final': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
    
    
        }
        return data_transforms


    #function to reads image data as well as class it belongs for categorical classification task
    def create_dataset(self):
        # map id to category
        test_items=[]
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        test_file=open(osp.join(self.root_dir, Config['test_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for x in test_file:
            test_items.append(x.strip('\n'))
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']

        # create X, y pairs
        files = os.listdir(self.image_dir)
        X = []; y = []
        for x in files:
            
            if x[:-4] in id_to_category:
                X.append(x)
                y.append(int(id_to_category[x[:-4]]))

        le=LabelEncoder()
        y=le.fit_transform(y)
        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1
    
    
    
    #function to read data files and lables for compatability testing task
    def create_compatability(self):
        X_compatTrain=[]
        Y_compatTrain=[]
        X_compatValid=[]
        Y_compatValid=[]
        train_comapatability=open(osp.join(self.root_dir, Config['train_compatability']), 'r')
        valid_comapatability=open(osp.join(self.root_dir, Config['valid_compatability']), 'r')
        for pair in train_comapatability:
            pair=pair.strip('\n')
            pair_list=pair.split()
            X_compatTrain.append((pair_list[1]+'.jpg',pair_list[2]+'.jpg'))
            Y_compatTrain.append(int(pair_list[0]))
        for pair in valid_comapatability:
            pair=pair.strip('\n')
            pair_list=pair.split()
            X_compatValid.append((pair_list[1]+'.jpg',pair_list[2]+'.jpg'))
            Y_compatValid.append(int(pair_list[0]))
            X_compatTrain
        return X_compatTrain,X_compatValid,Y_compatTrain,Y_compatValid
    

    
   
 
    

#creates data generator to train for categorical classification task
class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        X, y = np.stack(X), np.stack(y)
        return np.moveaxis(X, 1, 3), tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)


    def __data_generation(self, indexes):
        X = []; y = []
        for idx in indexes:
            file_path = osp.join(self.image_dir, self.X[idx])
            X.append(self.transform(Image.open(file_path)))
            y.append(self.y[idx])
        return X, y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    
    


#creates data generator for compatability testing task       
class DataGeneratorCompat(tensorflow.keras.utils.Sequence):
   
        
      
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = 2
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()



    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X1,X2, y = self.__data_generation(indexes)
        X1,X2,y = np.stack(X1), np.stack(X2),np.stack(y)
        return [np.moveaxis(X1, 1, 3), np.moveaxis(X2, 1, 3)], np.array(y)


    def __data_generation(self, indexes):
        X1 = []; X2 = [];y = []
        for idx in indexes:
            
            file_path1 = osp.join(self.image_dir, self.X[idx][0])
            file_path2 = osp.join(self.image_dir, self.X[idx][1])
            X1.append(self.transform(Image.open(file_path1)))
            X2.append(self.transform(Image.open(file_path2)))
            
            y.append(int(self.y[idx]))
        return X1,X2, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

