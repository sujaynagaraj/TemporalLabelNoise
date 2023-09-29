import sys
sys.path.insert(0,'..')

from src.data_gen import *
from src.noise import *
from src.logistic_regression import *
import pickle as pkl
import scipy

from sklearn.utils import shuffle

def save_data_real(X_clean, Y_clean, filename, path):
    with open(path+filename+".pkl",'wb') as f:
        pkl.dump([X_clean,Y_clean], f)

def load_data_real(filename, path):
    with open(path+filename+".pkl",'rb') as f:
        X_clean,Y_clean = pkl.load(f)
    return X_clean, Y_clean

def generate_filename_real(dataset_type, length, n_states):
    return f"{dataset_type}_length_{length}_n_states_{n_states}"

if __name__ == "__main__":

    #Process HAR Data
    dataset_type = "HAR"
    feature_ranges = [(1,9)]
    length = 100
    states = [2,4]

    for n_states in states:
        X_clean_train, Y_clean_train = load_HAR(feature_ranges, length, n_states, train=True)
        X_clean_test, Y_clean_test = load_HAR(feature_ranges, length, n_states, train=False)

        #Train
        filename = generate_filename_real(dataset_type, length, n_states)+"_train"
        path = "/data/real/processed/"
        save_data_real(X_clean_train, Y_clean_train, filename, path)

        #Test
        filename = generate_filename_real(dataset_type, length, n_states)+"_test"
        path = "/data/real/processed/"
        save_data_real(X_clean_test, Y_clean_test, filename, path)

    #Process EEG Eye Data
    dataset_type = "EEG_EYE"
    lengths = [25, 50, 100]
    n_states = 2

    for length in lengths:
        X_clean, Y_clean = load_EEG_EYE(length)
        filename = generate_filename_real(dataset_type, length, n_states)
        path = "/data/real/processed/"
        save_data_real(X_clean, Y_clean, filename, path)


    #Process EEG Sleep Data
    dataset_type = "EEG_SLEEP"
    minutes_around = 60
    lengths = [100, 200]
    states = [2,3]

    for n_states in states:
        for length in lengths:
            if n_states == 2:
                X_clean, Y_clean = load_EEG_SLEEP(minutes_around, length, binary=True)
            else:
                X_clean, Y_clean = load_EEG_SLEEP(minutes_around, length, binary=False)

            filename = generate_filename_real(dataset_type, length, n_states)
            path = "/data/real/processed/"
            save_data_real(X_clean, Y_clean, filename, path)

    

    #Process HAR70 Data
    dataset_type = "HAR70"
    length = 100
    n_states = 2

    X_clean, Y_clean = load_HAR70(length)
    filename = generate_filename_real(dataset_type, length, n_states)
    path = "/data/real/processed/"
    save_data_real(X_clean, Y_clean, filename, path)

    