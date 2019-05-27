import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def load_signals(path):
    file = open(path,'rb')
    return pickle.load(file,fix_imports=True,encoding = 'latin1')
    
def get_signals_noise_ratio(signal_data):
    return sorted(list(set([k[1] for k,v in signal_data.items()])), reverse = False)

def get_modulation_schemes(signal_data):
    return list(set([k[0] for k,v in signal_data.items()]))

def stack_signals(signal_data):
    signals = []  # our input values
    signal_noise_ratios = [] # our output values (SNRs and modulation types)
    modulations = []
    
    for mod in get_modulation_schemes(signal_data):
        for snr in get_signals_noise_ratio(signal_data):
            
            sub_signals = signal_data[(mod, snr)]
            size = sub_signals.shape[0]
            
            signals.append(sub_signals)
            modulations.extend([mod] * size)
            signal_noise_ratios.extend([snr] * size)
            
    return {'signals': np.vstack(signals),
            'modulations': np.array(modulations),
            'signal_noise_ratios': np.array(signal_noise_ratios)} # stack the number of time windows in the first dimension (rows)

def generate_train_test_set(signals, modulations, signal_noise_ratios, test_size):
    mod_snr_tuples = [(mod, snr) for mod, snr in zip(modulations, signal_noise_ratios)]
    signals_indicies = np.arange(len(signal_noise_ratios))
    X_train_ind, X_test_ind, y_train, y_test = train_test_split(signals_indicies, 
                                                        mod_snr_tuples, 
                                                        test_size=test_size, 
                                                        stratify=mod_snr_tuples)
    y_train = np.array([y[0] for y in y_train])
    y_test = np.array([y[0] for y in y_test])

    X_train, X_test = signals[X_train_ind], signals[X_test_ind]
    
    return  X_train, X_test, y_train, y_test

# def get_normalize_trainset(signals, X_train_ind, X_test_ind):
      
#     X_train, X_test = signals[X_train_ind], signals[X_test_ind]
#     X_train_shape, X_test_shape = signals[X_train_ind].shape, signals[X_test_ind].shape

    
    
#     X_train = standardize.fit_transform(X_train.reshape(X_train_shape[0], -1)).reshape(X_train_shape)
#     X_test = standardize.transform(X_test.reshape(X_test_shape[0], -1)).reshape(X_test_shape)

#     return X_train, X_test
    

def generate_binary_labels(modulations, modulation_of_interest):
    return [1 if modulation == modulation_of_interest else 0 for modulation in modulations]