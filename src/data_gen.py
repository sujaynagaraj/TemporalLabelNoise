from src.noise import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

from hmmlearn import hmm
from torch.distributions import uniform
from sklearn.preprocessing import normalize, StandardScaler
from scipy.io.arff import loadarff 

import os
from os import listdir
from os.path import isfile, join

from tqdm.notebook import tqdm
import pandas as pd
import seaborn as sns
import numpy as np

import mne
from mne.datasets.sleep_physionet.age import fetch_data


#Generate Datasets using HMM

#Generate a random, normalized transition matrix given a number of states
def random_transmat(n_states):
    matrix = np.random.rand(n_states, n_states) #Uniform distribution between [0,1)
    return matrix/matrix.sum(axis=1)[:,None] #normalize so rows sum to 1

#Generate a random start probability vector 
def random_startprob(n_states):
    startprob = np.random.rand(n_states)
    return startprob/startprob.sum()

def random_means(n_states, n_features):
    return np.random.randint(10, size=(n_states,n_features))

#Generate a dataset using a random hmm given number of samples, number of states, number of features and length of each sample
def sample_hmm(n_samples, n_states, n_dims , length, means = None, covars = None, startprob = None, transmat = None):
    #GENERATING A MODEL
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")

    if type(startprob) is np.ndarray:
         model.startprob_ = startprob
    else:
        model.startprob_ = random_startprob(n_states)
       
    if type(transmat) is np.ndarray:
         model.transmat_ = transmat
    else:
        model.transmat_ = random_transmat(n_states)

    if type(means) is np.ndarray:
        model.means_ = means
    else:
        model.means_ = random_means(n_states, n_dims)

    if type(covars) is np.ndarray:
        model.covars_ = covars
    else:
        model.covars_ = 0.5*np.tile(np.identity(n_dims), (n_dims, 1, 1))


    #SAMPLING FROM MODEL and STORING IN NP Array
    dataset=[]
    states = []

    
    for i in range(n_samples):
        X, Z = model.sample(length)
        dataset.append(np.array(X))
        states.append(Z)

    dataset = np.stack(dataset)
    states = np.stack(states)
    
    return dataset, states


def generate_dataset(n_samples, n_states, n_dims, length, dataset_type):
    means = []

    for i in range(n_states):
        means.append([])

    for i in range(n_dims):
        for k in range(n_states):
            mean = k
            #mean = np.random.randint(-n_states*5,n_states*5)
            #for k2 in range(n_states):
                #while mean in means[k2]:
                #    mean = np.random.randint(-n_states*5,n_states*5)
            means[k].append(mean)

    means = np.array(means)

    startprob = np.array(np.repeat(0.5, n_states)) #Equal likelihood of state 0 or state 1
    startprob = normalize(startprob.reshape(-1, 1), axis=0, norm='l1')
    startprob = startprob.reshape(1,-1)[0]

    transmat = np.tile(0.5, (n_states,n_states))
    transmat = normalize(transmat, axis=0, norm='l1')

    if dataset_type == "low_var":
        covars = 0.1*np.tile(np.identity(n_dims), (n_states, 1, 1))
    elif dataset_type == "med_var":
        covars = 0.5*np.tile(np.identity(n_dims), (n_states, 1, 1))
    elif dataset_type == "high_var":
        covars = 1.0*np.tile(np.identity(n_dims), (n_states, 1, 1))

    startprob = np.array(np.repeat(0.5, n_states)) #Equal likelihood of state 0 or state 1
    startprob = normalize(startprob.reshape(-1, 1), axis=0, norm='l1')

    startprob = startprob.reshape(1,-1)[0]

    transmat = np.tile(0.5, (n_states,n_states))
    transmat = normalize(transmat, axis=0, norm='l1')
    
    #Generate Data
    dataset, states_true = sample_hmm(n_samples, n_states, n_dims , length, means=means, covars=covars, startprob=startprob, transmat=transmat)
    
    return dataset, states_true

# def add_noise(dataset, states_true , method, flip_probability= None, flip_probability_0=None, flip_probability_1=None,
#                     noise_startprob=None, noise_transmat=None, a = None, b=None, num_samples=None, 
#                     sig_flip=False, center=20):
#     if method == "basic":
#         states_flipped = []
#         probabilities = []
#         for item in states_true:
#             states_flipped.append(flip_labels_basic(item, flip_probability=flip_probability))
#             probabilities.append(np.repeat(flip_probability,len(item)))

#     elif method == "class":
#         states_flipped = []
#         probabilities = []
#         for item in states_true:
#             states_flipped.append(flip_labels_class(item, flip_probability_0, flip_probability_1))
#             probs = np.copy(item)
#             probs[probs == 0] = flip_probability_0
#             probs[probs == 1] = flip_probability_1

#             probabilities.append(probs)

#     elif method == 'time':
#         markov_chain = generate_noise_markov_chain(noise_startprob, noise_transmat)
#         flip_probability = noise_transmat[0,1]
#         stay_probability = noise_transmat[1,1]
#         pi = get_stationary_distribution(flip_probability, stay_probability)

#         states_flipped = []
#         probabilities = []
#         #iterating over entire dataset of true states
#         for item in states_true:
#             flipped = flip_labels_time(item, markov_chain)
#             states_flipped.append(flipped)
#             probabilities.append(np.repeat(pi[1],len(item)))
            
#     elif method == "exp":
#         states_flipped = []
#         probabilities = []
#         #iterating over entire dataset of true states
#         for item in states_true:
#             flipped = flip_labels_exp(item, a, b, num_samples)
#             states_flipped.append(flipped)
#             probabilities.append(exponential_decay(a, b ,num_samples))

#     elif method == "sig":
#         states_flipped = []
#         probabilities = []
#         #iterating over entire dataset of true states
#         for item in states_true:
#             flipped = flip_labels_sig(item, a, b, num_samples, sig_flip, center)
#             states_flipped.append(flipped)
#             probabilities.append(sigmoid(a, b ,num_samples, sig_flip, center))

#     elif method == "lin":
#         states_flipped = []
#         probabilities = []
#         #iterating over entire dataset of true states
#         for item in states_true:
#             flipped = flip_labels_lin(item, a,b, num_samples)
#             states_flipped.append(flipped)
#             probabilities.append(np.linspace(a, b, num_samples))

#     elif method == "sin":
#         states_flipped = []
#         probabilities = []
#         #iterating over entire dataset of true states
#         for item in states_true:
#             flipped = flip_labels_sin(item, a, b, num_samples)
#             states_flipped.append(flipped)
#             probabilities.append(sin(a, b ,num_samples))


#     return dataset.astype("float"), states_true.astype("float"), np.array(states_flipped).astype("float"), np.array(states_true != states_flipped).astype(int), np.array(probabilities).astype("float")


def add_noise(dataset, states_true , method, num_classes, a = 0.49, b=0.01, c=0, mix_a = 0.49 , mix_b = 0.1, mix_c = 0, variant = "class_independent"):
    states_flipped = []
    mask = []
    T_t = T_t_generate(method, num_classes, states_true.shape[1], a, b,c, mix_a, mix_b, mix_c, variant = variant)
    for item in states_true:
        flipped, flip_mask = flip_labels_T_t(item, T_t)
        states_flipped.append(flipped)
        mask.append(flip_mask)
            
    return dataset.astype("float"), states_true.astype("float"), np.array(states_flipped).astype("float"), np.array(mask).astype(int), T_t

def set_hmm_parameters(dataset_type, n_dims, n_states):
    #Setting emissions
    if dataset_type == "equal_var_low": #Different means, equal variance (0.1 unit)

        ## [[mean_0_state_0, mean_1_state_0], [[mean_0_state_1, mean_1_state_1]]
        means = []
        
        for i in range(n_states):
            means.append([])
        
        for i in range(n_dims):
            for k in range(n_states):
                mean = np.random.randint(-5,5)
                for k2 in range(n_states):
                    while mean in means[k2]:
                        mean = np.random.randint(-5,5)
                means[k].append(mean)
            
        means = np.array(means)
        covars = 0.1*np.tile(np.identity(n_features), (2, 1, 1))

    elif dataset_type == "equal_var_high": #Different means, equal variance (2 unit)

        ## [[mean_0_state_0, mean_1_state_0], [[mean_0_state_1, mean_1_state_1]]
        means = []
        
        for i in range(n_states):
            means.append([])
        
        for i in range(n_dims):
            for k in range(n_states):
                mean = np.random.randint(-5,5)
                for k2 in range(n_states):
                    while mean in means[k2]:
                        mean = np.random.randint(-5,5)
                means[k].append(mean)
            
        means = np.array(means)
        covars = 2.0*np.tile(np.identity(n_dims), (2, 1, 1))
        
    elif dataset_type == "equal_mean": #Different vars, equal mean

        ## [[mean_0_state_0, mean_1_state_0], [[mean_0_state_1, mean_1_state_1]]
        means_state_0 = []
        means_state_1 = []
        mean = np.random.randint(-5,5)
        for i in range(n_dims):
            means_state_0.append(mean)
            means_state_1.append(mean)
        means = np.array([means_state_0, means_state_1])

        ## [[var_0_state_0, var_1_state_0], [[var_0_state_1, var_1_state_1]]
        vars_state_0 = (np.random.randint(1,5))

        #NEED TO ENSURE THE VARS ARE NOT IN vars_state_0
        var = np.random.randint(1,5)
        while var == vars_state_0:
            var = np.random.randint(1,5)
        vars_state_1=(var)

        covars = np.array([vars_state_0, vars_state_1])
        covars_state_0 = vars_state_0*np.identity(n_dims)
        covars_state_1 = vars_state_1*np.identity(n_dims)

        covars = np.array([covars_state_0, covars_state_1])

    startprob = np.array(np.repeat(0.5, n_states)) #Equal likelihood of state 0 or state 1
    startprob = normalize(startprob.reshape(-1, 1), axis=0, norm='l1')

    startprob = startprob.reshape(1,-1)

    transmat = np.tile(0.5, (n_states,n_states))
    transmat = normalize(transmat, axis=0, norm='l1')

    return means, covars, startprob, transmat

def generate_t_from_flip(flip_probability):
    return np.array([[1-flip_probability, flip_probability],[flip_probability, 1-flip_probability]])

def load_HAR70(length):

    PATH = "/data/real/har70plus/"

    features = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
    
    recoding = {1:0, 3:0, 4:0, 5:0, 6:1, 7:1, 8:1}


    X = []
    Y = []
    for filename in os.listdir(PATH):
        f = os.path.join(PATH, filename)
        # checking if it is a file

        if os.path.isfile(f):
            
            df = pd.read_csv(f)
            df = df.assign(label  = df.label.map(recoding))
            df['Datetime'] = pd.to_datetime(df['timestamp'])
            df = df.set_index("Datetime")
            #df = df.resample('10S').mean()
            
            #downsampling to 1Hz, by taking every 50th value
            df = df.iloc[::50, :]

            df = df.dropna()
            
            #N = len(df)//length #number of chunks 
            

            #normalized
            df[features] = StandardScaler().fit_transform(df[features])

            #split = np.array_split(df, N)
            split = [df[i:i+length] for i in range(1, len(df)-length, length)]
            
            for item in split:
                if len(item) == length:
                    X.append(item[features].values)
                    Y.append(item.label.values)

    return np.array(X), np.array(Y)


def load_EEG_EYE(length):

    recoding = {b'0': 0, b'1': 1}

    DATAPATH = "/data/real/EEG_EYE/EEG Eye State.arff"

    raw_data = loadarff(DATAPATH)
    df = pd.DataFrame(raw_data[0])

    df = df.assign(label  = df.eyeDetection.map(recoding))

    #N = len(df)//length #number of chunks 

    #truncated = df[:N*length]

    cols_to_norm = ["AF3", "F7", "F3", "FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]

    # detect and remove outliers
    z_scores = scipy.stats.zscore(df[cols_to_norm])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 5).all(axis=1)
    df_ro = df[filtered_entries]
    # reset index
    df_ro = df_ro.reset_index(drop=True)


    #normalized
    df_ro[cols_to_norm] = StandardScaler().fit_transform(df_ro[cols_to_norm])

    split = [df_ro[i:i+length] for i in range(1, len(df_ro)-length, length)]

    X = []
    Y = []
    for item in split:
        Y.append(item.label.values)
        X.append(item.drop(labels = ["eyeDetection", "label"], axis=1).values)

    return np.array(X), np.array(Y)

def load_HAR(feature_ranges, length, n_classes = 2, train=True):
    
    recoding_2 = {1:0, 2:0, 3:0, 4:1, 5:1, 6:1}
    recoding_4 = {1:0, 2:0, 3:0, 4:1, 5:2, 6:3}

    
    DATAPATH = "/data/real/UCI_HAR/"
    
    if train:
        SPLIT = "train/"
        END = "_train.txt"
    else:
        SPLIT = "test/"
        END = "_test.txt"
        
    RAW = "Inertial Signals/"


    # get the data from txt files to pandas dataframe
    X = pd.read_csv(DATAPATH+SPLIT+"X"+END, delim_whitespace=True, header=None)
    #X_train.columns = features

    Y = pd.read_csv(DATAPATH+SPLIT+"y"+END, delim_whitespace=True, header=None)

    subject = pd.read_csv(DATAPATH+SPLIT+"subject"+END, delim_whitespace=True, header=None)

    merged_df = X.copy()
    merged_df["label"] = Y.values
    merged_df["subject"] = subject.values
    
    if n_classes == 2:
        merged_df = merged_df.assign(label  = merged_df.label.map(recoding_2))
    elif n_classes == 4:
        merged_df = merged_df.assign(label  = merged_df.label.map(recoding_4))
        
    
    features = []

    for r in feature_ranges:
        for i in range(r[0],r[1]+1):
            features.append(i)

    X = []
    Y = []

    subjects = merged_df.subject.unique()

    for subject in subjects:
        sub_df = merged_df[merged_df["subject"]==subject]
        
        #N = len(sub_df)//length #number of chunks 
    
        #truncated = sub_df[:N*length]
        
        #normalized
        sub_df[features] = StandardScaler().fit_transform(sub_df[features])
        
        split = [sub_df[i:i+length] for i in range(1, len(sub_df)-length, length)]
        
        for item in split:
            X.append(item[features].values)
            Y.append(item.label.values)
    
    return np.array(X), np.array(Y)


def get_data_file(subject):
    PATH = "/data/real/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    
    for file in files:
        if subject in file and "PSG" in file:
            return file
def get_labels_file(subject):
    PATH = "/data/real/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    
    for file in files:
        if subject in file and "Hypnogram" in file:
            return file

def load_subject(subject):
    PATH = "/data/real/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    
    DATA_PATH = PATH + get_data_file(subject)
    LABELS_PATH = PATH + get_labels_file(subject)

    data = mne.io.read_raw_edf(
        DATA_PATH, stim_channel="Event marker"
    )

    labels = mne.read_annotations(LABELS_PATH)

    data.set_annotations(labels, emit_warning=False)
    
    return data, labels

def process_subject(data, labels, minutes_around = 60, length = 100, binary = True):
    
    if binary:
        recoding = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 1,
        "Sleep stage 3": 1,
        "Sleep stage 4": 1,
        "Sleep stage R": 1
        }
    else:
        recoding = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 1,
        "Sleep stage 3": 1,
        "Sleep stage 4": 1,
        "Sleep stage R": 2
        }

    data.set_annotations(labels, emit_warning=False)
    
    #Truncate to a first 60 min of wake before and after sleep
    first = int((labels[1]["onset"]- minutes_around * 60) * 100)
    second = int((labels[-2]["onset"] + minutes_around * 60) * 100)

    chunk_duration = 0.01#in seconds

    events_train, _ = mne.events_from_annotations(
        data, event_id=recoding, chunk_duration=chunk_duration
    )

    df = data.to_data_frame()
    df = df.drop(columns = ["time"])

    labels = events_train[:,2]
    
    #Padding zeros to the end to ensure data and labels are same length
    labels = np.pad(labels, (0, len(df)-len(labels)), 'constant')

    #adding labels column
    df["labels"] = labels

    
    df = df.iloc[first : second+1000]

    downsample = 60 #seconds to downsample to
    df = df.iloc[::downsample*100, :]
    
            
    features = df.drop(columns = ["labels"]).columns
    
    #normalized
    df[features] = StandardScaler().fit_transform(df[features])

    #split DataFrame into chunks
    split = [df[i:i+length] for i in range(0,len(df),length)]
    
    X = []
    Y = []
    for item in split:
        if len(item) == length:
            X.append(item[features].values)
            Y.append(item.labels.values)

    return np.array(X), np.array(Y)

def load_EEG_SLEEP(minutes_around, length, binary):
    PATH = "/data/real/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    subjects = [f[:6] for f in listdir(PATH) if isfile(join(PATH, f)) and "SC" in f]
    subjects = list(set(subjects))
    subjects.sort()
    
    X = []
    Y = []
    
    for subject in tqdm(subjects):
        try:
            data, labels = load_subject(subject)

            X_sub, Y_sub = process_subject(data, labels, minutes_around, length, binary)
            for item in X_sub:
                X.append(item)
            for item in Y_sub:
                Y.append(item)
        except:
            continue
    
    return np.array(X), np.array(Y)

