import sys
sys.path.insert(0,'..')

from src.data_gen import *
from src.noise import *
from src.logistic_regression import *
import pickle as pkl

import numpy as np

from sklearn.utils import shuffle

def save_data(X_clean, Y_clean, filename, path):
    with open(path+filename+".pkl",'wb') as f:
        pkl.dump([X_clean,Y_clean], f)

def load_data(filename, path):
    with open(path+filename+".pkl",'rb') as f:
        X_clean,Y_clean = pkl.load(f)
    return X_clean, Y_clean

def generate_filename(n_dims, n_states, n_samples, dataset_type, time_dependency):
    return f"{dataset_type}_{time_dependency}_n_states_{n_states}_n_dims_{n_dims}_n_samples_{n_samples}"

def train_test_split(X, Y, Y_tilde, mask):
    X, Y, Y_tilde, mask = shuffle(X, Y, Y_tilde, mask,  random_state=0)

    x_train = X[:int(0.8*len(X))]
    y_train = Y[:int(0.8*len(Y))]
    y_tilde_train = Y_tilde[:int(0.8*len(Y_tilde))]
    mask_train = mask[:int(0.8*len(mask))]
    
    x_test = X[int(0.8*len(X)):]
    y_test = Y[int(0.8*len(Y)):]
    y_tilde_test = Y_tilde[int(0.8*len(Y_tilde)):]
    mask_test = mask[int(0.8*len(mask)):]


    return x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test

def train_val_test_split(X, Y, Y_tilde, mask):
    X, Y, Y_tilde, mask = shuffle(X, Y, Y_tilde, mask,  random_state=0)

    x_train = X[:int(0.6*len(X))]
    y_train = Y[:int(0.6*len(Y))]
    y_tilde_train = Y_tilde[:int(0.6*len(Y_tilde))]
    mask_train = mask[:int(0.6*len(mask))]

    x_val = X[int(0.6*len(X)):int(0.8*len(X))]
    y_val = Y[int(0.6*len(Y)):int(0.8*len(Y))]
    y_tilde_val = Y_tilde[int(0.6*len(Y_tilde)):int(0.8*len(Y_tilde))]
    mask_val = mask[int(0.6*len(mask)):int(0.8*len(mask))]
    
    x_test = X[int(0.8*len(X)):]
    y_test = Y[int(0.8*len(Y)):]
    y_tilde_test = Y_tilde[int(0.8*len(Y_tilde)):]
    mask_test = mask[int(0.8*len(mask)):]


    return x_train, y_train, y_tilde_train, mask_train, x_val, y_val, y_tilde_val, mask_val, x_test, y_test, y_tilde_test, mask_test

if __name__ == "__main__":
    #clean time-series
    #clean time-series
    states = [2, 3]
    length = 100
    samples = [1000]
    dims = [15]
    dataset = ["high_var"]
    time_dependencies = ["none"]

    for n_states in tqdm(states):
        for n_dims in tqdm(dims):
            for n_samples in samples:
                for dataset_type in dataset:
                    for time_dependency in time_dependencies:

                        X_clean, Y_clean = generate_dataset(n_samples, n_states, n_dims, length, dataset_type)

                        time = (0.1*(np.arange(1,length+1)))
                        time = np.expand_dims(time, 1)

                        if time_dependency == "add":
                            X_clean = X_clean + time
                        elif time_dependency == "mult":
                            X_clean = np.multiply(X_clean, time)
                        elif time_dependency == "both":
                            X_clean = np.multiply(X_clean+time, time)

                        filename = generate_filename(n_dims, n_states, n_samples, dataset_type, time_dependency)
                        path = "/data/clean/"+time_dependency+"/"

                        save_data(X_clean, Y_clean, filename, path)