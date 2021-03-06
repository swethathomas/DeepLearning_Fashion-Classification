from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from data import polyvore_dataset, DataGenerator
from utils import Config

import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
import numpy as np
from tensorflow.keras.utils import plot_model



if __name__=='__main__':

    #create data generators
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
 


    # Use GPU
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
        
    #use pretrained mobile net for classification
    base_model = MobileNet(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(n_classes, activation = 'softmax')(x)
    
    #define model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    # define optimizers
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    plot_model(model,to_file='DLQ1P1_model.png',show_shapes=True,show_layer_names=True)



    # training
    results=model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        workers=Config['num_workers'],
                        epochs=Config['num_epochs']
                        )

    #save weights
    model.save('polyvore_trained.hdf5')
    loss=results.history['loss']
    acc=results.history['accuracy']
    val_acc=results.history['val_accuracy']
    
    np.savetxt('accuracy.txt',acc)
    np.savetxt('val_acc',val_acc)
    

