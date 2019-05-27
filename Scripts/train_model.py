import os
os.chdir('../Python/')

import urllib, tarfile
import numpy as np
import pandas as pd

from signaldiscovery.etl import stack_signals, load_signals, get_modulation_schemes, get_signals_noise_ratio, \
generate_train_test_set

from signaldiscovery.models import IndependentModulationClassifier


def pull_data():

    success = False

    print('Retreiving data from deepsig.')
    url = "http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2"
    save_path = "../Data/RML2016.10a.tar.bz2"
    urllib.request.urlretrieve(url, save_path)
    print('Data Retreived')

    print('Extracting data from compressed file.')
    tar = tarfile.open(save_path, "r:bz2")
    tar.extractall(path="../Data/",)
    tar.close()
    print('Data extracted from compressed file.')

    if os.path.exists(save_path):
        print('Deleting compressed file')
        os.remove(save_path)
        success =True
        print('Deleting compressed file')
        print('Download process complete')
    else:
        print("File not downloaded", end='\r')

    return success

if not 'RML2016.10a_dict.pkl' in os.listdir('../Data/'):
    pull_data()
    
signal_data = load_signals('../Data/RML2016.10a_dict.pkl')
signal_dict = stack_signals(signal_data)
unique_modulations = get_modulation_schemes(signal_data)
unique_signal_noise_ratios = get_signals_noise_ratio(signal_data)


num_modulations = NUM_CLASSES = len(unique_modulations)
num_signals = len(unique_signal_noise_ratios)
num_of_samples = num_modulations * num_signals * 1000

print('Number of modulations: {}'.format(num_modulations))
print('Number of signal-to-noise: {}'.format(num_signals))
print('Total number of signals in the dataset: {} '.format(num_of_samples))
print('Number of Samples per modulation scheme: {}'.format(int(num_of_samples/num_modulations)))
print('Proportion of samples for each modulation: {}'.format(20000/num_of_samples))


signals = signal_dict['signals']
modulations = signal_dict['modulations']
signal_noise_ratios = signal_dict['signal_noise_ratios']

#convert modulation schemes to binary lables


#generate train & test sets
train_data, test_data, train_labels, test_labels = generate_train_test_set(signals, 
                                                                           modulations,
                                                                           signal_noise_ratios, 
                                                                           test_size=0.3)

#train the model 
INPUT_SHAPE = [2, 128]
model = IndependentModulationClassifier([2, 128])
model.fit(train_data, train_labels, weights_path='../Data/models', epochs=30, batch_size=512)

#save the model 
model.save_model('../Data/models')