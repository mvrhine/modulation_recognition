from sklearn.preprocessing import StandardScaler
import numpy as np

import pickle

from signaldiscovery import metrics
from signaldiscovery.etl import generate_binary_labels


from keras.models import Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

class IndependentModulationClassifier:
    
    def __init__(self, input_shape):
    
        self.input_shape = input_shape 
        self.model_dict = {}
        self.unique_modulations = None
        self._scaler = StandardScaler()
        
    def generate_ind_model(self):
        """ 
        Build and compile Keras Convolutional Neural Network model object. 

        Returns: 
        keras Squential obj: Keras CNN object containing 4 convolutional layers

        """
        
        dr = 0.5 # dropout rate (%) = percentage of neurons to randomly lose each iteration
        model = Sequential()  # Neural network is a set of sequential layers
        model.add(Reshape(self.input_shape + [1], input_shape = self.input_shape))
        model.add(Conv2D(256, (1,3), activation="relu"))
        model.add(Dropout(dr))
        model.add(Conv2D(80, (1,3), activation="relu"))
        model.add(Dropout(dr))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(dr))
        model.add(Dense(1,activation='sigmoid', kernel_initializer='he_normal'))
        model.compile(loss=metrics.f1_loss, optimizer=Adam(lr=0.001), metrics=['accuracy', metrics.fbeta,  metrics.precision,  metrics.recall])
        return model
    
    def fit(self, train_data, train_labels, weights_path='', epochs=100, batch_size=1024, validation_size=0.2):
        
        training_output = {}
        self.unique_labels = np.unique(train_labels)

        train_data_norm = self._scaler.fit_transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        
        for step, label in enumerate(self.unique_labels):
            
            print("Modulation {} out of {} \n".format(step + 1, len(self.unique_labels)), end='\n')

            #convert modulation schemes to binary lables
            binary_labels = generate_binary_labels(train_labels, label)

            #initialize model object
            model = self.generate_ind_model()

            #Weight the positive class to counteract class-imbalance
            class_weight = {1: 1 / (np.sum(binary_labels) / len(binary_labels)), 0:1}
            
            #set path for model weights
            if weights_path:
                weights_path += '/'
            filepath = weights_path + 'ind_cnn_weights_' + label + '_0.5.wts.h5'
        
            #initialize model call backs and train the model
            callbacks = [
                ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
            ]

            training_output[label] = model.fit(train_data_norm,
                                                    binary_labels,
                                                    epochs=epochs,
                                                    batch_size=batch_size,
                                                    class_weight=class_weight, 
                                                    validation_split=validation_size, 
                                                    callbacks=callbacks)
            
            model.load_weights(filepath)
            
            self.model_dict[label] = model
            

        return training_output
    
    def load_weights(self, path):
        for label in self.unique_labels:
            filepath = path + '/ind_cnn_weights_' + label + '_0.5.wts.h5'
            temp_model = self.generate_ind_model()
            temp_model.load_weights(filepath)
            self.model_dict[label] = temp_model
        return None
    
    def load_model(self, path):

        with open(path + '/labels.pkl', 'rb') as aFile:
            self.unique_labels = pickle.load(aFile)
        with open(path + '/scaler.pkl', 'rb') as aFile:
            self._scaler  = pickle.load(aFile)

        self.load_weights(path)

        return None

    def save_model(self, path):

        with open(path + '/labels.pkl', 'wb') as aFile:
            pickle.dump(self.unique_labels, aFile)
        with open(path + '/scaler.pkl', 'wb') as aFile:
            pickle.dump(self._scaler, aFile)

        for label in self.unique_labels:
            filepath = path + '/ind_cnn_weights_' + label + '_0.5.wts.h5'
            self.model_dict[label].save_weights(filepath)

        return None


    def predict_classes(self, dataset):
        label_map = {i:label for i,label in enumerate(self.unique_labels)}
        dataset_norm = self._scaler.transform(dataset.reshape(dataset.shape[0], -1)).reshape(dataset.shape)
        probabilities = self.predict_probas(dataset_norm)

        indicies = np.argmax(probabilities, axis=1)
        y_pred = np.array([label_map.get(ind, np.nan) for ind in indicies])
        return y_pred

    def predict_probas(self, dataset):
        
        probabilities = []

        #Fix array shape if dataset contains 1 recored
        if len(dataset.shape) == 2:
            dataset.reshape(1, dataset.shape[0], dataset.shape[1])

        for modulation in self.unique_labels:
            probabilities.append(self.model_dict[modulation].predict_proba(dataset))

        return np.hstack(probabilities)