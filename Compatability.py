from tensorflow.keras.layers import GlobalAveragePooling2D, Dense,BatchNormalization,Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

from data import polyvore_dataset, DataGeneratorCompat
from utils import Config

import tensorflow as tf
import numpy as np

if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test = dataset.create_compatability()
    

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'shuffle': True
              }
    
    
    
    train_generator =  DataGeneratorCompat(train_set, dataset_size, params)
    test_generator = DataGeneratorCompat(test_set, dataset_size, params)
    
   
   
        
   
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_shape=(224,224,3)
        
        left_input = Input(input_shape)
        right_input = Input(input_shape)
            
        model = models.Sequential()
        model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D(3,3))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D(5,5))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        
        model.add(layers.Conv2D(128, (3,3), activation='relu'))
        model.add(layers.Flatten())
       
        model.add(layers.Dropout(0.2))
        model.add(Dense(512,activation='sigmoid'))
        
        encoded_l = model(left_input)
        encoded_r = model(right_input)
        
        
        L1_layer = layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        
        
        prediction = Dense(1,activation='sigmoid')(L1_distance)
        
        
        siamese_network = Model(inputs=[left_input,right_input],outputs=prediction)
        siamese_network.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        
        siamese_network.summary()
        plot_model(model,to_file='DLQ2_model.png',show_shapes=True,show_layer_names=True)
       
    results = siamese_network.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=False,
                        workers=Config['num_workers'],
                        epochs = Config['num_epochs']
                        )
    
    
    siamese_network.save('polyvore_trainedQuestion3.hdf5')
    
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']
    
    np.savetxt('acc_Question3.txt',acc)
    np.savetxt('val_accQuestion3.txt',val_acc)


    
