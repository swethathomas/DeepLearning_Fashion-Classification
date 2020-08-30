 from tensorflow.keras.layers import GlobalAveragePooling2D, Dense,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model

from data import polyvore_dataset, DataGenerator
from utils import Config

import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
import numpy as np



if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator =  DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)
    
    
    
    
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    #design custom model architecture 
    model = models.Sequential()
        # BLOCK 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
        # BLOCK 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
        # BLOCK 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
           
    model.add(layers.MaxPooling2D((2, 2)))
        # BLOCK 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
           
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
        
    model.add(layers.Dropout(0.2))
        
   
       
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))
    
    #compile model
    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    

    

    model.summary()
    
    plot_model(model,to_file='DLQ1P2_model.png',show_shapes=True,show_layer_names=True)



    # training
    results=model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        workers=Config['num_workers'],
                        epochs=Config['num_epochs']
                        )

    #save model weights
    model.save('polyvore_trained.hdf5')
    loss=results.history['loss']
    acc=results.history['accuracy']
    val_acc=results.history['val_accuracy']
    
    np.savetxt('accuracy.txt',acc)
    np.savetxt('val_acc',val_acc)
    


