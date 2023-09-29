from src.data_gen import *
import torch
import torch.nn as nn
import numpy as np


def generate_noise_markov_chain(startprob, transmat):
    #GENERATING A MODEL
    markov_chain = hmm.GaussianHMM(n_components=2, covariance_type="full")
    markov_chain.startprob_ = startprob
    markov_chain.transmat_ = transmat
    #print(markov_chain.startprob_)
    #this doesn't actually matter for us
    markov_chain.means_ = np.array([[0.0, 0.0], 
                             [5.0, 10.0]])
    markov_chain.covars_ = np.tile(np.identity(2), (3, 1, 1))

    
    #ignoring emissions, we only care about the markov chain of the latent states
    #X, Z = markov_chain.sample(length)

    return markov_chain

#returns stationary distribution of a markov chain
#pi = [p()]
#noise_transmat = np.array([[(1-flip_probability), flip_probability],
#                          [ (1-stay_probability), stay_probability]])
def get_stationary_distribution(flip_probability, stay_probability):
#noise_transmat = np.array([[(1-flip_probability), flip_probability],
    pi = np.array([(1-stay_probability)/((1-stay_probability)+flip_probability), (flip_probability)/((1-stay_probability)+flip_probability)])
    return pi

#Injecting Noise into Labels

#Given a flip_mask, flip an input
def flip(array, flip_mask):
    flipped_array = np.logical_xor(array, flip_mask)
    return flipped_array

#Class Independent / Time Independent
def flip_labels_basic(array, flip_probability):
    flip_mask = np.random.binomial(1, flip_probability, len(array))
    return flip(array, flip_mask)

#Class Dependent / Time Independent
def flip_labels_class(array, flip_probability_0, flip_probability_1):
    flip_mask = []
    for elem in array:
        if elem == 0:
            to_flip = np.random.binomial(1, flip_probability_0, 1)[0]
            flip_mask.append(to_flip)
        else:
            to_flip = np.random.binomial(1, flip_probability_1, 1)[0]
            flip_mask.append(to_flip)
            
    return flip(array, flip_mask)

#Class Independent / Time Dependent
def flip_labels_time(array, model):
    _, flip_mask = model.sample(len(array))

    return flip(array, flip_mask)


#Non-stationary time dependent noise
def exponential_decay(a, b, N):
    # a, b: exponential decay parameter
    # N: number of samples 
    return a * (1-b*(100/N)) ** np.arange(N)

#plt.plot(exponential_decay(0.99, 0.05, 100))

def sigmoid(a,b,c = 0, N = 100, sig_flip=False):
    #a: upper bound on probability
    #b: steepness
    #N: number of samples
    
    x = np.linspace(0, N, N)
    if sig_flip:
        return ((a-c)/(1 + np.exp((x*b*(100/N))-(N/2)*b*(100/N)))+c)
    else:
        return ((a-c)/(1 + np.exp((-x*b*(100/N))+(N/2)*b*(100/N)))+c)

def sin(a,b,c,N):
    #a: frequency
    #b: max amplitude
    #N: number of samples
    
    x = np.linspace(0, N, N)
    return (((b-c)/2*np.sin(a*(100/N)*x))+((b-c)/2))+c

#Class Independent / Time Dependent
def flip_labels_exp(array, a, b, N):
    #flip probabilities over time, exponentially decaying to 0
    flip_probabilities = exponential_decay(a,b ,N)
    #binary flip mask based on flip_probabilities
    flip_mask = np.random.binomial(1, flip_probabilities)

    return flip(array, flip_mask)

#Class Independent / Time Dependent
def flip_labels_sig(array, a, b, c=0, N = 100, sig_flip=False):
    #flip probabilities over time, exponentially decaying to 0
    flip_probabilities = sigmoid(a,b,c ,N, sig_flip)
    #binary flip mask based on flip_probabilities
    flip_mask = np.random.binomial(1, flip_probabilities)

    return flip(array, flip_mask)


#Class Independent / Time Dependent
def flip_labels_sin(array, a, b,c=0, N=100):
    #flip probabilities over time, using a sin function
    flip_probabilities = sin(a,b ,c, N)
    #binary flip mask based on flip_probabilities
    flip_mask = np.random.binomial(1, flip_probabilities)

    return flip(array, flip_mask)

#def lin_decay(a, N, start = 0.99):
#    lin_decay =  np.linspace(start, 0.0, a)
#    flip_probabilities = np.pad(lin_decay,(0,N-len(lin_decay)), "constant", constant_values=0.0)
#    return flip_probabilities

def flip_labels_lin(array, a, b, N):
    #flip probabilities over time, starting at a linearly decaying to b
    flip_probabilities = np.linspace(a, b, N)
    #binary flip mask based on flip_probabilities
    flip_mask = np.random.binomial(1, flip_probabilities)
    return flip(array, flip_mask)

#Given all mask vectors, calculate average number of flips in dataset
def empirical_flip_frequency(mask):
    if mask.ndim == 1:
        return (len(mask[mask==1])/len(mask))
    else:
        lis = []
        for m in mask:
            lis.append(len(m[m==1])/len(m))
        return np.average(lis)

#Get flip fliquency from T_t, but taking the mean over off diagonal entries
def get_flip_frequency(T_t):
    return np.mean([a[~np.eye(a.shape[0],dtype=bool)].sum()/a.shape[0] for a in T_t])

#given noisy trained model logits, estimate anchor points in dataset across time (one anchor point per class)
#Outputs in row-stochastic form
def estimate_anchor(predictions, n_states, quantile=None):
    time_steps = predictions.shape[1]
    
    P = np.tile(np.ones((n_states, n_states)), (time_steps,1,1))
    
    flattened_predictions = torch.flatten(predictions, 0,1)

    for t in range(time_steps):
        for c_i in range(n_states):
            
            if quantile == None:
                anchor_point = flattened_predictions[torch.max(flattened_predictions[:,c_i], 0)[1],:]
            else:
                anchor_point = flattened_predictions[torch.topk(flattened_predictions[:,c_i], int((1-quantile)*len(flattened_predictions[:,c_i])))[1][-1],:]

            for c_j in range(n_states):
                P[t,c_i, c_j] = anchor_point[c_j]
    return P

#given noisy trained model logits, estimate noise rates at each time step (one anchor point per timestep per class)
#Outputs in row-stochastic format
def estimate_anchor_time(predictions, n_states, quantile=None):
    time_steps = predictions.shape[1]
    
    P = np.tile(np.ones((n_states, n_states)), (time_steps,1,1))

    for t in range(time_steps):
        for c_i in range(n_states):
            if quantile == None:
                anchor_point = predictions[torch.max(predictions[:,t,c_i], 0)[1],t,:]
            else:
                anchor_point = predictions[torch.topk(predictions[:,t,c_i], int((1-quantile)*len(predictions[:,t,c_i])))[1][-1],t,:]
            for c_j in range(n_states):
                P[t,c_i, c_j] = anchor_point[c_j]
    return P




# def transition_matrix_generate(noise_rate=0.5, num_classes=10):
#     P = np.ones((num_classes, num_classes))
#     n = noise_rate
#     P = (n / (num_classes - 1)) * P
    
#     if n > 0.0:
#         # 0 -> 1
#         P[0, 0] = 1. - n
#         for i in range(1, num_classes-1):
#             P[i, i] = 1. - n

#         P[num_classes-1, num_classes-1] = 1. - n

#     return P

# def T_t_generate(noise_type, num_classes, time_steps, a, b):
#     P = np.tile(np.ones((num_classes, num_classes)), (time_steps,1,1))
    
#     if noise_type == "basic":
#         return np.tile(transition_matrix_generate(noise_rate=a, num_classes = num_classes), (time_steps, 1,1))
#     else:
        
#         if noise_type == "exp":
#             flip_probabilities = exponential_decay(a,b ,time_steps)
#         elif noise_type == "sig":
#             flip_probabilities = sigmoid(a,b,time_steps, center=20)
#         elif noise_type == "sin":
#             flip_probabilities = sin(a,b, time_steps)
#         elif noise_type == "lin":
#             flip_probabilities = np.linspace(a, b, 100)
            
#         flip_probabilities_spread = (flip_probabilities / (num_classes - 1))

#         for i in range(0, num_classes):
#             for j in range(0, num_classes):
#                 if i !=j:
#                     P[:,i,j] = flip_probabilities_spread

#         for i in range(0, num_classes):
#             for j in range(0, num_classes):
#                 if i == j:
#                     P[:,i,j] = 1-flip_probabilities

#         return P

def transition_matrix_generate(noise_rate=0.4, num_classes=2, variant="class_independent"):
    if variant == "class_conditional":
        n = {}
        P = np.ones((num_classes, num_classes))
        for i in range(num_classes):
            n[i] = noise_rate / (0.25*i+1)
            P[i] = (n[i] / (num_classes - 1)) * P[i]

        for i in range(num_classes):
            P[i, i] = 1. - n[i]
            
    else: #class independent
        P = np.ones((num_classes, num_classes))
        n = noise_rate
        P = (n / (num_classes - 1)) * P

        P[0, 0] = 1. - n
        for i in range(1, num_classes-1):
            P[i, i] = 1. - n

        P[num_classes-1, num_classes-1] = 1. - n

    return P

def T_t_generate(noise_type, num_classes, time_steps, a, b, c = 0, mix_a = 0.49, mix_b = 0.1, mix_c =0, variant = "class_independent"):
    
    if noise_type == "basic":
        return np.tile(transition_matrix_generate(noise_rate=a, 
                                                num_classes = num_classes, 
                                                variant = variant), (time_steps, 1,1))

    else:

        P = np.tile(np.ones((num_classes, num_classes)), (time_steps,1,1))
        n = {}
        spread = {}
        
        for i in range(num_classes):
        
            if noise_type == "exp":
                n[i] = exponential_decay(a,b ,time_steps)
            elif noise_type == "sig":
                n[i] = sigmoid(a,b,c,time_steps)
            elif noise_type == "sin":
                n[i] = sin(a,b,c, time_steps)
            elif noise_type == "lin":
                n[i] = np.linspace(a, b, time_steps)
            elif noise_type == "mix":
                if i+1 <= num_classes/2:
                    n[i] = exponential_decay(a,b ,time_steps)
                else:
                    n[i] = sigmoid(mix_a,mix_b, mix_c,time_steps)
            
        for i in range(num_classes):
            if variant == "class_conditional":
                n[i] = n[i] / (0.25*i+1)
            spread[i] = (n[i] / (num_classes - 1))
            
        for i in range(0, num_classes):
            for j in range(0, num_classes):
                if i !=j:
                    P[:,i,j] = spread[i]

        for i in range(0, num_classes):
            for j in range(0, num_classes):
                if i == j:
                    P[:,i,j] = 1-n[i]

        return P
    


def flip_labels_T_t(array, T_t):
    time_steps = T_t.shape[0]

    flipped = np.ones(time_steps)
    
    for t in range(time_steps):
        flipped[t] = (np.argmax(np.random.multinomial(1, T_t[t,array[t],:]), axis = 0))
        
    flip_mask = 1*~np.equal(array, flipped)
    return flipped.astype("int"), flip_mask