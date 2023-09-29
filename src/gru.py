from src.data_gen import *
from src.noise import *
from src.loss_functions import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from copy import deepcopy

from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import time
from tqdm.notebook import tqdm
import pandas as pd
import seaborn as sns
import scipy

from torch.optim.lr_scheduler import MultiStepLR


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        #self.to(self.device)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        #print(out.shape)
        out =(self.fc((out)))
        #out =(self.fc((out)))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    A_init = sampler.sample((n_units, n_units))[..., 0]  
    return A_init

class GRUNet_noisy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, add_noise_tensor, drop_prob=0.0, add_noise=0.0, proportional_noise=False):
        super(GRUNet_noisy, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.add_noise = add_noise
        self.add_noise_tensor = add_noise_tensor
        self.proportional_noise = proportional_noise
        #self.to(self.device)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        if self.proportional_noise:
            add_noise = self.add_noise_tensor.unsqueeze(1).repeat_interleave(self.hidden_dim, dim=1).float().to(self.device) * torch.randn(out.shape[0], out.shape[1], out.shape[2]).float().to(self.device)
        else:
            add_noise = self.add_noise * torch.randn(out.shape[0], out.shape[1], out.shape[2]).float().to(self.device)
        out =(self.fc((out+add_noise)))
        #out =(self.fc((out)))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden

class NoisyRNN(nn.Module):
    def __init__(self, input_dim, output_classes, device, n_units=8, eps=0.01, 
                 beta=0.8, gamma_A=0.01, gamma_W=0.01, init_std=1, alpha=1,
                 solver='noisy', add_noise=0, mult_noise=0):
        super(NoisyRNN, self).__init__()

        
        #modified
        self.device = device

        self.n_units = n_units
        self.eps = eps
        self.solver = solver
        self.beta = beta
        self.alpha = alpha
        self.gamma_A = gamma_A
        self.gamma_W = gamma_W
        self.add_noise = add_noise
        self.mult_noise = mult_noise
        
        self.tanh = nn.Tanh()

        self.E = nn.Linear(input_dim, n_units)
        self.D = nn.Linear(n_units, output_classes)     
                                            
        self.C = nn.Parameter(gaussian_init_(n_units, std=init_std))            
        self.B = nn.Parameter(gaussian_init_(n_units, std=init_std))    
        self.I = torch.eye(n_units).to(self.device)   

        self.d = nn.Parameter(torch.rand(self.n_units).float().to(self.device)*0 + eps)           


    def forward(self, x, mode='test'):
        T = x.shape[1]
        h = torch.zeros(x.shape[0], self.n_units).to(self.device)
        
        hidden_states = []
        for i in range(T):
            z = self.E(x[:,i,:])

            if i == 0:
                    A = self.beta * (self.B - self.B.transpose(1, 0)) + (1-self.beta) * (self.B + self.B.transpose(1, 0)) - self.gamma_A * self.I
                    W = self.beta * (self.C - self.C.transpose(1, 0)) + (1-self.beta) * (self.C + self.C.transpose(1, 0)) - self.gamma_W * self.I
                
            add_noise = 0.0
            mult_noise = 1.0
            if mode == 'train':
                if self.add_noise > 0:
                    add_noise = self.add_noise * torch.randn(h.shape[0], h.shape[1]).float().to(self.device)
                            
                if self.mult_noise > 0:
                    #mult_noise = self.mult_noise * torch.randn(h.shape[0], h.shape[1]).float().to(self.device) + 1
                    mult_noise = self.mult_noise * torch.rand(h.shape[0], h.shape[1]).float().to(self.device) + (1-self.mult_noise)
                        

            if self.solver == 'base': 
                h_update = self.alpha * torch.matmul(h, A) + self.tanh(torch.matmul(h, W) + z)                
                h = h + self.eps * h_update
            elif self.solver == 'noisy':
                h_update = self.alpha * torch.matmul(h, A) + self.tanh(torch.matmul(h, W) + z)                
                h = h + self.d * mult_noise * h_update + add_noise                              
            hidden_states.append(h)
            
        hidden_states = torch.stack(hidden_states)
        hidden_states = torch.transpose(hidden_states, 0,1)
        #print(hidden_states.shape)
        # Decoder 
        #----------
        #out = self.D(h)
        out = self.D(hidden_states)
        return out

class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t, self).__init__()

        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(num_classes, num_classes)))

        self.w.to(device)

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device)
        self.identity = torch.eye(num_classes).to(device)

    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=0)

        return T

class sig_t2(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t2, self).__init__()

        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(num_classes, num_classes)))

        self.w.to(device)

        self.identity = torch.eye(num_classes).to(device)

    def forward(self):

        sig = torch.softmax(self.w, dim = -2)
        T = 0.5*(self.identity.detach() + sig)

        return T

def checkpoint(model, PATH):
    torch.save(model.state_dict(), PATH)

def train_RNN(n_features, train_loader, loss_function, learning_rate, output_dim=1, n_layers = 1, hidden_dim=32, EPOCHS=150, model_type="GRU", noise_probability=None, verbose=True, keep_metrics = True, add_noise=0.05, add_noise_tensor=None, proportional_noise=False):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim
    
    # Instantiating the model
    if model_type == "NoisyRNN":
        model = NoisyRNN(input_dim, output_dim, n_units=hidden_dim, device=device, add_noise=add_noise)
    elif model_type == "NoisyGRU":
        model = GRUNet_noisy(input_dim, hidden_dim, output_dim, n_layers, device=device, add_noise=add_noise, add_noise_tensor=add_noise_tensor, proportional_noise=proportional_noise)
    else: #GRU
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)

    # Defining loss function and optimizer
    if loss_function == "backward_method":
        criterion = backward_method().to(device)
    elif loss_function == "noise_regularized_loss":
        criterion = noise_regularized_loss().to(device)
    elif loss_function == "forward_method":
        criterion = forward_method().to(device)
    elif loss_function == "forward_method_time":
        criterion = forward_method_time().to(device)
    elif loss_function == "backward_method_time":
        criterion = backward_method_time().to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    if verbose:
        print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    losses = []
    losses_clean = []
    losses_noisy = []

    fractions_correct = []
    fractions_incorrect = []

    fractions_correct_noisy = []
    fractions_incorrect_noisy = []
    fractions_memorized = []
    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()

        avg_loss = 0.
        avg_loss_clean = 0.
        avg_loss_noisy = 0.

        avg_fractions_correct = 0.
        avg_fractions_incorrect = 0.

        avg_fractions_correct_noisy = 0.
        avg_fractions_incorrect_noisy = 0.
        avg_fractions_memorized = 0.

        counter = 0
        #start_time = time.clock()
        for x, label, truth, mask, probabilities in train_loader:
            if model_type == "NoisyRNN":
                counter+=1
                model.zero_grad()
                out = model(x.to(device).float(), "train")
                out = torch.squeeze(out, -1)
            else:
                h = model.init_hidden(x.shape[0])
                counter += 1
                model.zero_grad()

                out, h = model(x.to(device).float(), h)
                out = torch.squeeze(out, -1)
            
            #Flattened across batches
            #predictions = torch.tensor([0 if torch.sigmoid(value) <= 0.5 else 1 for value in torch.flatten(out.cpu().detach())])
            predictions = torch.round(torch.sigmoid(torch.flatten(out.cpu())))
            mask = torch.flatten(mask)
            truth = torch.flatten(truth)
        
            if loss_function == "backward_method" or loss_function == "forward_method":
                #Batch Loss
                loss = criterion(out, label.to(device).float(), torch.as_tensor(noise_probability).to(device).float())
                if keep_metrics:
                    out = torch.flatten(out)
                    label = torch.flatten(label)
                    loss_clean = criterion(out[mask==0], label[mask==0].to(device).float(), torch.as_tensor(noise_probability).to(device).float())
                    loss_noisy = criterion(out[mask==1], label[mask==1].to(device).float(), torch.as_tensor(noise_probability).to(device).float())
            elif loss_function == "noise_regularized_loss" or loss_function == "backward_method_time" or loss_function =="forward_method_time":

                loss = criterion(out, label.to(device).float(), probabilities.to(device).float())
                if keep_metrics:
                    out = torch.flatten(out)
                    label = torch.flatten(label)
                    probs = torch.flatten(probabilities)
                    loss_clean = criterion(out[mask==0], label[mask==0].to(device).float(), probs[mask==0].to(device).float())
                    loss_noisy = criterion(out[mask==1], label[mask==1].to(device).float(), probs[mask==1].to(device).float())
            else: #BCE Loss
                #Batch Loss
                loss = criterion(out, label.to(device).float())
                
                if keep_metrics:
                    out = torch.flatten(out)
                    label = torch.flatten(label)
                    loss_clean = criterion(out[mask==0], label[mask==0].to(device).float())
                    loss_noisy = criterion(out[mask==1], label[mask==1].to(device).float())
                   
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if keep_metrics:
                #Clean Labels
                fraction_correct = len(predictions[mask==0][predictions[mask==0] == truth[mask==0]])/len(predictions[mask==0])
                fraction_incorrect = len(predictions[mask==0][predictions[mask==0] != truth[mask==0]])/len(predictions[mask==0])

                #Noisy Labels
                if noise_probability!=0.0:
                    fraction_correct_noisy = len(predictions[mask==1][predictions[mask==1] == truth[mask==1]])/len(predictions[mask==1])
                    fraction_incorrect_noisy = len(predictions[mask==1][predictions[mask==1] != truth[mask==1]])/len(predictions[mask==1])
                    fraction_memorized = len(predictions[mask==1][predictions[mask==1] == label[mask==1]])/len(predictions[mask==1])

                avg_loss_clean += loss_clean.item()
                avg_loss_noisy += loss_noisy.item()

                avg_fractions_correct += fraction_correct
                avg_fractions_incorrect += fraction_incorrect

                if noise_probability!=0.0:
                    avg_fractions_memorized +=fraction_memorized
                    avg_fractions_correct_noisy += fraction_correct_noisy
                    avg_fractions_incorrect_noisy += fraction_incorrect_noisy
            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        current_time = time.clock()
        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
    
        #Per Epoch Metrics
        losses.append(avg_loss/len(train_loader))
        
        epoch_times.append(current_time-start_time)
        if keep_metrics:
            losses_clean.append(avg_loss_clean/len(train_loader))
            losses_noisy.append(avg_loss_noisy/len(train_loader))

            fractions_correct.append(avg_fractions_correct/len(train_loader))
            fractions_incorrect.append(avg_fractions_incorrect/len(train_loader))

            fractions_correct_noisy.append(avg_fractions_correct_noisy/len(train_loader))
            fractions_incorrect_noisy.append(avg_fractions_incorrect_noisy/len(train_loader))
            fractions_memorized.append(avg_fractions_memorized/len(train_loader))

            
    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    
    return model, losses, losses_clean, losses_noisy, fractions_correct, fractions_incorrect, fractions_correct_noisy, fractions_incorrect_noisy, fractions_memorized


def train_RNN_volmin(n_features, train_loader, experimentID, learning_rate, learning_rate_trans = 0.01, output_dim=2, n_layers = 1, lam = 0.01, hidden_dim=32, EPOCHS=150, verbose=True, early_stopping = False, x_val=None, y_tilde_val = None, opt_trans = "adam", milestones = [30, 60], milestones_trans = [30, 60], gamma = 0.1, gamma_trans=0.1):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim
    
    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)
    
    #Initialize T generating weights
    trans = sig_t(device, output_dim)
    trans.to(device)

    t = trans()
    trans.train()
    
    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if opt_trans == "adam":
        optimizer_trans = torch.optim.Adam(trans.parameters(), lr=learning_rate_trans)
    else:
        optimizer_trans = torch.optim.SGD(trans.parameters(), lr=learning_rate_trans, momentum=0.9, weight_decay = 0.001)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scheduler_trans = MultiStepLR(optimizer_trans, milestones=milestones_trans, gamma=gamma_trans)

    model.train()
    if verbose:
        print("Starting Training of RNN Vol Min model")
    epoch_times = []
    # Start training loop
    
    early_stop_thresh = 10
    best_val_loss = 1000
    best_epoch = -1

    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()

        avg_loss = 0.

        counter = 0

        #start_time = time.clock()
        for x, label, _, _ in train_loader:

            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            optimizer_trans.zero_grad()
            t = trans()
            
            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(t,clean_posterior.unsqueeze(-1))

            ce_loss = criterion(torch.flatten(noisy_posterior.squeeze().log(), 0, 1), label.flatten().to(device).long())

            vol_loss = t.slogdet().logabsdet
            #vol_loss = t.det()
            loss = ce_loss + lam*vol_loss
            
                   
            loss.backward()
            optimizer.step()
            optimizer_trans.step()

            avg_loss += loss.item()
            
            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        current_time = time.clock()
        scheduler.step()
        scheduler_trans.step()
            
        
        if early_stopping:
             #VALIDATION LOSS
            h = model.init_hidden(x_val.shape[0])
            model.zero_grad()

            out, h = model(x_val.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            optimizer_trans.zero_grad()
            t = trans()
            
            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(t,clean_posterior.unsqueeze(-1))

            ce_loss = criterion(torch.flatten(noisy_posterior.squeeze().log(), 0, 1), y_tilde_val.flatten().to(device).long())
            
            vol_loss = t.slogdet().logabsdet
            #vol_loss = t.det()
            val_loss = ce_loss + lam*vol_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(model, "checkpoints/"+experimentID+".pth")
                
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop

        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Avg Loss: {}, Best Val Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader), best_val_loss))
        
        epoch_times.append(current_time-start_time)
        
    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    est_T = t.detach().cpu()

    #Return Row Stochastic form
    est_T = np.transpose(est_T)

    if early_stopping:
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
        model.load_state_dict(torch.load("checkpoints/"+experimentID+".pth"))
        model.eval()

    return model, est_T

def train_RNN_volmin2(n_features, train_loader, experimentID, learning_rate, learning_rate_trans = 0.01, output_dim=2, n_layers = 1, lam = 0.01, hidden_dim=32, EPOCHS=150, verbose=True, early_stopping = False, x_val=None, y_tilde_val = None, opt_trans = "adam", milestones = [30, 60], milestones_trans = [30, 60], gamma = 0.1, gamma_trans=0.1):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim
    
    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)
    
    #Initialize T generating weights
    trans = sig_t2(device, output_dim)
    trans.to(device)

    t = trans()
    trans.train()
    
    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if opt_trans == "adam":
        optimizer_trans = torch.optim.Adam(trans.parameters(), lr=learning_rate_trans)
    else:
        optimizer_trans = torch.optim.SGD(trans.parameters(), lr=learning_rate_trans, momentum=0.9, weight_decay = 0.001)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scheduler_trans = MultiStepLR(optimizer_trans, milestones=milestones_trans, gamma=gamma_trans)

    model.train()
    if verbose:
        print("Starting Training of RNN Vol Min model")
    epoch_times = []
    # Start training loop
    
    early_stop_thresh = 10
    best_val_loss = 1000
    best_epoch = -1

    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()

        avg_loss = 0.

        counter = 0

        #start_time = time.clock()
        for x, label, _, _ in train_loader:

            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            optimizer_trans.zero_grad()
            t = trans()
            
            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(t,clean_posterior.unsqueeze(-1))

            ce_loss = criterion(torch.flatten(noisy_posterior.squeeze().log(), 0, 1), label.flatten().to(device).long())

            vol_loss = t.slogdet().logabsdet
            #vol_loss = t.det()
            loss = ce_loss + lam*vol_loss
            
                   
            loss.backward()
            optimizer.step()
            optimizer_trans.step()

            avg_loss += loss.item()
            
            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        current_time = time.clock()
        scheduler.step()
        scheduler_trans.step()
            
        
        if early_stopping:
             #VALIDATION LOSS
            h = model.init_hidden(x_val.shape[0])
            model.zero_grad()

            out, h = model(x_val.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            optimizer_trans.zero_grad()
            t = trans()
            
            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(t,clean_posterior.unsqueeze(-1))

            ce_loss = criterion(torch.flatten(noisy_posterior.squeeze().log(), 0, 1), y_tilde_val.flatten().to(device).long())
            
            vol_loss = t.slogdet().logabsdet
            #vol_loss = t.det()
            val_loss = ce_loss + lam*vol_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(model, "checkpoints/"+experimentID+".pth")
                
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop

        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Avg Loss: {}, Best Val Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader), best_val_loss))
        
        epoch_times.append(current_time-start_time)
        
    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    est_T = t.detach().cpu()

    #Return Row Stochastic form
    est_T = np.transpose(est_T)

    if early_stopping:
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
        model.load_state_dict(torch.load("checkpoints/"+experimentID+".pth"))
        model.eval()

    return model, est_T


def train_RNN_anchor(n_features, train_loader, experimentID, learning_rate, output_dim=2, n_layers = 1, hidden_dim=32, EPOCHS=150, verbose=True, keep_metrics = True, early_stopping=False,  x_val = None, y_tilde_val = None, quantile=None, opt = "adam", milestones = [50,100], gamma = 0.1):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim
    
    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)

    # Defining loss function and optimizer
    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)
    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 0.001)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    model.train()

    epoch_times = []

    # Start training loop
    losses = []
    noise_estimates = []

    early_stop_thresh = 10
    best_val_loss = 1000
    best_epoch = -1

    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()
        avg_loss = 0.

        counter = 0
        #start_time = time.clock()
        
        for x, label, _, _ in train_loader:
            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            predictions = softmax(out)
            
            loss = criterion(predictions.flatten(end_dim=1).squeeze().log(), label.flatten().to(device).long())

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            current_time = time.clock()
        
        current_time = time.clock()
        scheduler.step()
       
        if early_stopping:
             #VALIDATION LOSS
            h = model.init_hidden(x_val.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x_val.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            predictions = softmax(out)
            
            val_loss = criterion(predictions.flatten(end_dim=1).squeeze().log(), y_tilde_val.flatten().to(device).long())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(model, "checkpoints/"+experimentID+".pth")
                
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop

        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Avg Loss: {}, Best Val Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader), best_val_loss))
        #Per Epoch Metrics
        losses.append(avg_loss/len(train_loader))
        
        epoch_times.append(current_time-start_time)

    if early_stopping:
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
        model.load_state_dict(torch.load("checkpoints/"+experimentID+".pth"))
        model.eval()
        model.to(device)
    
    all_predictions = []

    #Get Final Noise estimates across entire dataset
    for x, label, _, _ in train_loader:

        model.eval()
        h = model.init_hidden(x.shape[0])

        out, h = model(x.to(device).float(), h)
        out = torch.squeeze(out, -1)
        
        predictions = softmax(out)
        
        all_predictions.append(predictions)

    all_predictions = torch.cat(all_predictions)
    final_noise_estimates = estimate_anchor(all_predictions, output_dim, quantile=quantile)

    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))


    return model, losses, noise_estimates, final_noise_estimates

def train_RNN_anchor_warmup(n_features, train_loader, experimentID, learning_rate, output_dim=2, n_layers = 1, hidden_dim=32, EPOCHS=150, warmup = 50, verbose=True, keep_metrics = True, early_stopping=False, x_val = None, y_tilde_val = None, quantile = None, opt = "adam", milestones = [50,100], gamma=0.1):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim
    
    time_steps = next(iter(train_loader))[0].shape[1]
    
    # Instantiating the model
    warmup_model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    warmup_model.to(device)

    # Defining loss function and optimizer
    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)

    optimizer = torch.optim.Adam(warmup_model.parameters(), lr=learning_rate)

    warmup_model.train()

    early_stop_thresh = 10
    best_val_loss = 1000
    best_epoch = -1
    
    #WARMUP LOOP to ESTIMATE ANCHOR POINTS
    for epoch in range(1,warmup+1):
        start_time = time.clock()

        for x, label, _, _ in train_loader:
            h = warmup_model.init_hidden(x.shape[0])
            warmup_model.zero_grad()

            out, _ = warmup_model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)
    
            predictions = softmax(out)
            
            loss = criterion(torch.flatten(predictions.squeeze().log(), 0, 1), label.flatten().to(device).long())
            
            loss.backward()
            optimizer.step()
            
    all_predictions = []
    #Get Final Noise estimates across entire dataset
    for x, label, _, _ in train_loader:

        warmup_model.eval()
        h = warmup_model.init_hidden(x.shape[0])

        out, _ = warmup_model(x.to(device).float(), h)
        out = torch.squeeze(out, -1)
        
        predictions = softmax(out)
        
        all_predictions.append(predictions)

    all_predictions = torch.cat(all_predictions)
    T_t = torch.from_numpy(estimate_anchor(all_predictions, output_dim, quantile=quantile)).float().to(device)
    
    #transpose to column stochastic
    T_t = torch.transpose(T_t, -1, -2)

    epoch_times = []

    # Start training loop
    losses = []
    noise_estimates = []
    

    # Instantiating the second model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
    model.to(device)
    
    # Defining loss function and optimizer
    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)

    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 0.001)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    model.train()
    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()
        avg_loss = 0.
        counter = 0
        
        for x, label, _, _ in train_loader:
            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(T_t, clean_posterior.unsqueeze(-1)).squeeze()

            loss = criterion(noisy_posterior.log().flatten(end_dim=1), label.flatten().to(device).long())

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        
        current_time = time.clock()
        scheduler.step()

        if early_stopping:
            #VALIDATION LOSS
            h = model.init_hidden(x_val.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x_val.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(T_t, clean_posterior.unsqueeze(-1)).squeeze()

            val_loss = criterion(noisy_posterior.log().flatten(end_dim=1), y_tilde_val.flatten().to(device).long())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(model, "checkpoints/"+experimentID+".pth")
                
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop
        

        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Avg Loss: {}, Best Val Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader), best_val_loss))
        #Per Epoch Metrics
        losses.append(avg_loss/len(train_loader))
        
        epoch_times.append(current_time-start_time)


    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    if early_stopping:
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
        model.load_state_dict(torch.load("checkpoints/"+experimentID+".pth"))
        model.eval()
        model.to(device)
    
    #return to row-stochastic form for visualization
    T_t = torch.transpose(T_t, -1, -2).cpu().numpy()
    
    return model, losses, noise_estimates, T_t


def train_RNN_anchor_time(n_features, train_loader, experimentID, learning_rate, output_dim=2, n_layers = 1, hidden_dim=32, EPOCHS=150, verbose=True, keep_metrics = True, early_stopping=False, x_val = None, y_tilde_val = None, quantile=None, opt = "adam", milestones = [50, 100], gamma = 0.1):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim
    
    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)

    # Defining loss function and optimizer
    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)

    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 0.001)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    model.train()

    epoch_times = []

    # Start training loop
    losses = []
    noise_estimates = []

    early_stop_thresh = 10
    best_val_loss = 1000
    best_epoch = -1
    
    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()
        avg_loss = 0.

        counter = 0
        #start_time = time.clock()
        
        for x, label, _, _ in train_loader:
            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            #Flattened across batches
            #predictions = torch.tensor([0 if torch.sigmoid(value) <= 0.5 else 1 for value in torch.flatten(out.cpu().detach())])
            predictions = softmax(out)
            
            loss = criterion(predictions.flatten(end_dim=1).squeeze().log(), label.flatten().to(device).long())

            #noise_estimate = estimate_noise_time(predictions, output_dim)
            #noise_estimates.append(noise_estimate)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item()

            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        
        current_time = time.clock()

    
        if early_stopping:
            #VALIDATION LOSS
            h = model.init_hidden(x_val.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x_val.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            #Flattened across batches
            #predictions = torch.tensor([0 if torch.sigmoid(value) <= 0.5 else 1 for value in torch.flatten(out.cpu().detach())])
            predictions = softmax(out)
            
            val_loss = criterion(predictions.flatten(end_dim=1).squeeze().log(), y_tilde_val.flatten().to(device).long())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(model, "checkpoints/"+experimentID+".pth")
                
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop
            
        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Avg Loss: {}, Best Val Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader), best_val_loss))
        #Per Epoch Metrics
        losses.append(avg_loss/len(train_loader))
        
        epoch_times.append(current_time-start_time)

    if early_stopping:
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
        model.load_state_dict(torch.load("checkpoints/"+experimentID+".pth"))
        model.eval()
        model.to(device)

    all_predictions = []

    #Get Final Noise estimates across entire dataset
    for x, label, _, _ in train_loader:

        model.eval()
        h = model.init_hidden(x.shape[0])

        out, h = model(x.to(device).float(), h)
        out = torch.squeeze(out, -1)
        
        predictions = softmax(out)
        
        all_predictions.append(predictions)

    all_predictions = torch.cat(all_predictions)
    final_noise_estimates = estimate_anchor_time(all_predictions, output_dim, quantile=quantile)

    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))


    return model, losses, noise_estimates, final_noise_estimates

def train_RNN_anchor_time_warmup(n_features, train_loader, experimentID, learning_rate, output_dim=2, n_layers = 1, hidden_dim=32, EPOCHS=150, warmup = 50, verbose=True, keep_metrics = True, early_stopping=False,  x_val = None, y_tilde_val = None, quantile=None, opt = "adam", milestones = [50,100], gamma=0.1):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim
    
    time_steps = next(iter(train_loader))[0].shape[1]
    
    # Instantiating the model
    warmup_model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    warmup_model.to(device)

    # Defining loss function and optimizer
    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)

    optimizer = torch.optim.Adam(warmup_model.parameters(), lr=learning_rate)

    warmup_model.train()
    
    #WARMUP LOOP to ESTIMATE ANCHOR POINTS
    for epoch in range(1,warmup+1):
        start_time = time.clock()

        
        for x, label, _, _ in train_loader:
            h = warmup_model.init_hidden(x.shape[0])
            warmup_model.zero_grad()

            out, _ = warmup_model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)
    
            predictions = softmax(out)
            
            loss = criterion(torch.flatten(predictions.squeeze().log(), 0, 1), label.flatten().to(device).long())
            
            loss.backward()
            optimizer.step()
            
    all_predictions = []
    #Get Final Noise estimates across entire dataset
    for x, label, _, _ in train_loader:

        warmup_model.eval()
        h = warmup_model.init_hidden(x.shape[0])

        out, _ = warmup_model(x.to(device).float(), h)
        out = torch.squeeze(out, -1)
        
        predictions = softmax(out)
        
        all_predictions.append(predictions)

    all_predictions = torch.cat(all_predictions)
    T_t = torch.from_numpy(estimate_anchor_time(all_predictions, output_dim, quantile=quantile)).float().to(device)
    
    #putting to column stochastic format
    T_t = torch.transpose(T_t, -1, -2)
    T_t.detach()

    epoch_times = []

    # Start training loop
    losses = []
    noise_estimates = []
    
    # Instantiating the second model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
    model.to(device)
    
    # Defining loss function and optimizer
    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)

    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 0.001)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    model.train()

    early_stop_thresh = 10
    best_val_loss = 1000
    best_epoch = -1

    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()
        avg_loss = 0.
        counter = 0

        for x, label, _, _ in train_loader:
            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(T_t,clean_posterior.unsqueeze(-1)).squeeze()

            loss = criterion(noisy_posterior.log().flatten(end_dim=1), label.flatten().to(device).long())

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        
        current_time = time.clock()
        scheduler.step()

        if early_stopping:
            #VALIDATION LOSS
            h = model.init_hidden(x_val.shape[0])
            model.zero_grad()

            out, h = model(x_val.to(device).float(), h)
            out = torch.squeeze(out, -1)

            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(T_t,clean_posterior.unsqueeze(-1)).squeeze()

            val_loss = criterion(noisy_posterior.log().flatten(end_dim=1), y_tilde_val.flatten().to(device).long())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(model, "checkpoints/"+experimentID+".pth")
                
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop

        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Avg Loss: {}, Best Val Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader), best_val_loss))
    
        #Per Epoch Metrics
        losses.append(avg_loss/len(train_loader))
        
        epoch_times.append(current_time-start_time)


    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    if early_stopping:
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
        model.load_state_dict(torch.load("checkpoints/"+experimentID+".pth"))
        model.eval()
        model.to(device)
        
    #return to row-stochastic form for visualization
    T_t = torch.transpose(T_t, -1, -2).cpu().numpy()

    return model, losses, noise_estimates, T_t

def train_RNN_volmin_time(n_features, train_loader, experimentID, learning_rate, learning_rate_trans = 0.01, output_dim=2, n_layers = 1, lam = 0.01, lam_frob = 0.01, hidden_dim=32, EPOCHS=150, verbose=True, early_stopping = False, x_val = None, y_tilde_val = None, opt_trans = "adam", milestones = [30,60], milestones_trans = [30, 60], gamma=0.1, gamma_trans=0.1, vol_loss_type = "sum_log_det"):

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim

    #Get total number of time steps
    time_steps = next(iter(train_loader))[0].shape[1]

    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)

    #Initialize T generating weights
    trans_list = []
    optimizers_list = []
    schedulers_list = []
    for i in range(time_steps):
        trans = sig_t(device, output_dim)
        trans.to(device)

        t = trans()
        trans.train()
        trans_list.append(trans)

        if opt_trans == "adam":
            optimizer_trans = torch.optim.Adam(trans.parameters(), lr=learning_rate_trans)
        else:
            optimizer_trans = torch.optim.SGD(trans.parameters(), lr=learning_rate_trans, momentum=0.9, weight_decay = 0.001)
        
        scheduler_trans = MultiStepLR(optimizer_trans, milestones=milestones_trans, gamma=gamma_trans)

        optimizers_list.append(optimizer_trans)
        schedulers_list.append(scheduler_trans)


    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    model.train()
    if verbose:
        print("Starting Training of RNN VolMin Time model")
    epoch_times = []
    # Start training loop

    early_stop_thresh = 10
    best_val_loss = 1000
    best_epoch = -1

    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()

        avg_loss = 0.

        counter = 0
        #start_time = time.clock()

        for x, label, _, _ in train_loader:

            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)

            for t in range(time_steps):
                optimizers_list[t].zero_grad()


            T = []
            for t in range(time_steps):
                T.append(trans_list[t]())

            T = torch.stack(T)

            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(T,clean_posterior.unsqueeze(-1)).squeeze()

            ce_loss = criterion(noisy_posterior.log().flatten(end_dim=1), label.flatten().to(device).long())
            
            if vol_loss_type == "sum_det":
                vol_loss = torch.mean(T.det())/time_steps
            elif vol_loss_type == "sum_log_det":
                vol_loss =  torch.mean(T.slogdet().logabsdet)/time_steps
            elif vol_loss_type == "det_sum":
                vol_loss = torch.div(torch.sum(T, dim=0), time_steps).det()
            elif vol_loss_type == "log_det_sum":
                vol_loss = torch.div(torch.sum(T, dim=0), time_steps).slogdet().logabsdet

            identity = torch.from_numpy(np.tile(np.identity((output_dim)), (time_steps,1,1))).to(device)
            #difference = torch.sub(T, identity)
            had_prod = T * identity
            frobenius_norm = torch.mean(torch.norm(had_prod, 'fro', dim = (1,2)))

            loss = ce_loss + lam*(vol_loss) + lam_frob*frobenius_norm

            loss.backward()
            optimizer.step()

            for t in range(time_steps):
                optimizers_list[t].step()

            avg_loss += loss.item()

        current_time = time.clock()
        scheduler.step()
        for t in range(time_steps):
                schedulers_list[t].step()

        if early_stopping:
            #VALIDATION LOSS
            h = model.init_hidden(x_val.shape[0])
            model.zero_grad()

            out, h = model(x_val.to(device).float(), h)
            out = torch.squeeze(out, -1)


            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(T,clean_posterior.unsqueeze(-1)).squeeze()

            ce_loss = criterion(noisy_posterior.log().flatten(end_dim=1), y_tilde_val.flatten().to(device).long())
            vol_loss = torch.mean(T.det())

            val_loss = ce_loss + lam*vol_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(model, "checkpoints/"+experimentID+".pth")
                
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop

        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Avg Loss: {}, Best Val Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader), best_val_loss))

        epoch_times.append(current_time-start_time)


    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    if early_stopping:
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
        model.load_state_dict(torch.load("checkpoints/"+experimentID+".pth"))
        model.eval()
        model.to(device)

    return model, trans_list

def train_RNN_volmin_time2(n_features, train_loader, experimentID, learning_rate, learning_rate_trans = 0.01, output_dim=2, n_layers = 1, lam = 0.01, lam_frob = 0.01, hidden_dim=32, EPOCHS=150, verbose=True, early_stopping = False, x_val = None, y_tilde_val = None, opt_trans = "adam", milestones = [30,60], milestones_trans = [30, 60], gamma=0.1, gamma_trans=0.1, vol_loss_type = "sum_log_det"):

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim

    #Get total number of time steps
    time_steps = next(iter(train_loader))[0].shape[1]

    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)

    #Initialize T generating weights
    trans_list = []
    optimizers_list = []
    schedulers_list = []
    for i in range(time_steps):
        trans = sig_t2(device, output_dim)
        trans.to(device)

        t = trans()
        trans.train()
        trans_list.append(trans)

        if opt_trans == "adam":
            optimizer_trans = torch.optim.Adam(trans.parameters(), lr=learning_rate_trans)
        else:
            optimizer_trans = torch.optim.SGD(trans.parameters(), lr=learning_rate_trans, momentum=0.9, weight_decay = 0.001)
        
        scheduler_trans = MultiStepLR(optimizer_trans, milestones=milestones_trans, gamma=gamma_trans)

        optimizers_list.append(optimizer_trans)
        schedulers_list.append(scheduler_trans)


    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    model.train()
    if verbose:
        print("Starting Training of RNN VolMin Time model")
    epoch_times = []
    # Start training loop

    early_stop_thresh = 10
    best_val_loss = 1000
    best_epoch = -1

    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()

        avg_loss = 0.

        counter = 0
        #start_time = time.clock()

        for x, label, _, _ in train_loader:

            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)

            for t in range(time_steps):
                optimizers_list[t].zero_grad()


            T = []
            for t in range(time_steps):
                T.append(trans_list[t]())

            T = torch.stack(T)

            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(T,clean_posterior.unsqueeze(-1)).squeeze()

            ce_loss = criterion(noisy_posterior.log().flatten(end_dim=1), label.flatten().to(device).long())
            
            if vol_loss_type == "sum_det":
                vol_loss = torch.mean(T.det())/time_steps
            elif vol_loss_type == "sum_log_det":
                vol_loss =  torch.mean(T.slogdet().logabsdet)/time_steps
            elif vol_loss_type == "det_sum":
                vol_loss = torch.div(torch.sum(T, dim=0), time_steps).det()
            elif vol_loss_type == "log_det_sum":
                vol_loss = torch.div(torch.sum(T, dim=0), time_steps).slogdet().logabsdet

            identity = torch.from_numpy(np.tile(np.identity((output_dim)), (time_steps,1,1))).to(device)
            #difference = torch.sub(T, identity)
            had_prod = T * identity
            frobenius_norm = torch.mean(torch.norm(had_prod, 'fro', dim = (1,2)))

            loss = ce_loss + lam*(vol_loss) + lam_frob*frobenius_norm

            loss.backward()
            optimizer.step()

            for t in range(time_steps):
                optimizers_list[t].step()

            avg_loss += loss.item()

        current_time = time.clock()
        scheduler.step()
        for t in range(time_steps):
                schedulers_list[t].step()

        if early_stopping:
            #VALIDATION LOSS
            h = model.init_hidden(x_val.shape[0])
            model.zero_grad()

            out, h = model(x_val.to(device).float(), h)
            out = torch.squeeze(out, -1)


            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(T,clean_posterior.unsqueeze(-1)).squeeze()

            ce_loss = criterion(noisy_posterior.log().flatten(end_dim=1), y_tilde_val.flatten().to(device).long())
            vol_loss = torch.mean(T.det())

            val_loss = ce_loss + lam*vol_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(model, "checkpoints/"+experimentID+".pth")
                
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop

        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Avg Loss: {}, Best Val Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader), best_val_loss))

        epoch_times.append(current_time-start_time)


    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    if early_stopping:
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
        model.load_state_dict(torch.load("checkpoints/"+experimentID+".pth"))
        model.eval()
        model.to(device)

    return model, trans_list


def evaluate_RNN(model, x_test, y_test, model_type="GRU", forward_correct = False, T_t=None):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    outputs = []
    targets = []
    
    x_test = torch.from_numpy(x_test)
    softmax = nn.Softmax(dim=2)
    
    if model_type == "NoisyRNN":
        model.eval()
        model.to(device)
        out = model(x_test.to(device).float(), "test")
        out = out.squeeze(-1)
    else:
        model.eval()
        model.to(device)
        h = model.init_hidden(x_test.shape[0])
        out, h = model(x_test.to(device).float(), h.to(device).float())
        out = out.squeeze(-1)


    accuracies = []
    time_steps = y_test.shape[-1]

    if forward_correct:
        T_t = torch.from_numpy(T_t)

        for t in range(time_steps):
            noisy_posterior = softmax(out[:,t,:].unsqueeze(2))
            forward_corrected = torch.matmul(torch.t(T_t[t,:,:]).cpu().float(),noisy_posterior.cpu().float()).squeeze()
            accuracy = accuracy_score(torch.round(forward_corrected).argmax(dim=1), torch.from_numpy(y_test[:,t]))
            accuracies.append(accuracy)
    else:
        for i in range(len(out)):
            accuracy = accuracy_score(torch.round(softmax(out))[i].argmax(dim=1).cpu().detach(), torch.from_numpy(y_test[i]))
            accuracies.append(accuracy)

    return accuracies, np.mean(accuracies)


def evaluate_RNN_time(model, x_test, y_test):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    outputs = []
    targets = []
    
    x_test = torch.from_numpy(x_test)
    softmax = nn.Softmax(dim=2)

    model.eval()
    model.to(device)
    h = model.init_hidden(x_test.shape[0])
    out, h = model(x_test.to(device).float(), h.to(device).float())
    out = out.squeeze(-1)


    accuracies = []
    accuracies_time = []
    time_steps = y_test.shape[-1]


    for i in range(len(out)):
        accuracy = accuracy_score(torch.round(softmax(out))[i].argmax(dim=1).cpu().detach(), torch.from_numpy(y_test[i]))
        accuracies.append(accuracy)

    for t in range(time_steps):
    
        accuracy = accuracy_score(torch.round(softmax(out))[:,t,:].argmax(dim=1).cpu().detach(), torch.from_numpy(y_test[:,t]))
        accuracies_time.append(accuracy)

    return accuracies, np.mean(accuracies), accuracies_time


class T_t_gen(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(T_t_gen, self).__init__()
        
        self.num_classes = num_classes
        self.device = device
        self.epsilon = 1e-5

        #co = torch.ones(num_classes, num_classes, requires_grad = False)
        #ind = np.diag_indices(co.shape[0])
        #co[ind[0], ind[1]] = torch.zeros(co.shape[0], requires_grad = False)
        #self.co = co.to(device)
        self.identity = torch.eye(num_classes, requires_grad = False).to(device)
        
        self.stack = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes*num_classes),
        )


    def forward(self, t):
        #t = torch.tensor([float(t)]).to(self.device)
        entries = self.stack(t)
        entries = entries.reshape(t.shape[0], self.num_classes, self.num_classes)
    
        #sig = torch.sigmoid(entries)
        #T = self.identity.detach() + sig*self.co.detach()
        
        sig = torch.softmax(entries, dim = -2)
        T = 0.5*(self.identity.detach() + sig)
        
        # T = T + self.identity.detach()*self.epsilon + (self.co.detach()*(-1)*self.epsilon / (self.num_classes-1))
        #T = F.normalize(T, p=1, dim=-2)
        #print(T[1,:,:])
        return T
    
def init_weights(m):
    if type(m) == nn.Linear:
        #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(m.weight, gain=1.0)

# Define a function to initialize the weights and biases
def initialize_weights_and_biases(module):
    if isinstance(module, nn.Linear):
        # Initialize weights close to zero
        nn.init.normal_(module.weight, mean=0.0, std=0.1)
        # Initialize biases to zero
        nn.init.zeros_(module.bias)

import time
from torch.nn.utils import spectral_norm

    
def train_RNN_volmin_T_t(n_features, train_loader, experimentID, learning_rate, learning_rate_trans = 0.01, output_dim=2, n_layers = 1, lam = 0.01,lam_frob=0.01, hidden_dim=32, EPOCHS=150, verbose=True, early_stopping = False,  x_val = None, y_tilde_val = None, opt_trans="adam", milestones = [30,60], milestones_trans = [30, 60], gamma = 0.1, gamma_trans = 0.1, vol_loss_type = "sum_log_det", init = False):

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim

    #Get total number of time steps
    time_steps = next(iter(train_loader))[0].shape[1]

    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)

    #Initialize T generating network
    trans = T_t_gen(device, output_dim)
    if init:
        trans.apply(initialize_weights_and_biases)

    trans.to(device)
    
    trans.train()

    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if opt_trans == "adam":
        optimizer_trans = torch.optim.Adam(trans.parameters(), lr=learning_rate_trans)
    else:
        optimizer_trans = torch.optim.SGD(trans.parameters(), lr=learning_rate_trans, momentum=0.9, weight_decay = 0.001)
    

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scheduler_trans = MultiStepLR(optimizer_trans, milestones=milestones_trans, gamma=gamma_trans)

    model.train()
    if verbose:
        print("Starting Training of RNN VolMin Ours model")
    epoch_times = []
    # Start training loop
    early_stop_thresh = 10
    best_val_loss = 1000
    best_epoch = -1

    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()

        avg_loss = 0.

        counter = 0

        #start_time = time.clock()
        for x, label, _, _ in train_loader:

            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()
            optimizer_trans.zero_grad()

            out, _ = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)

            time_steps_tensor = torch.arange(time_steps, requires_grad=False).float().unsqueeze(-1).to(device)

            est_T = trans(time_steps_tensor)

            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(est_T,clean_posterior.unsqueeze(-1)).squeeze()

            ce_loss = criterion(noisy_posterior.log().flatten(end_dim=1), label.flatten().to(device).long())
            
            if vol_loss_type == "sum_det":
                vol_loss = torch.mean(est_T.det())/time_steps
            elif vol_loss_type == "sum_log_det":
                vol_loss =  torch.mean(est_T.slogdet().logabsdet)/time_steps
            elif vol_loss_type == "det_sum":
                vol_loss = torch.div(torch.sum(est_T, dim=0), time_steps).det()
            elif vol_loss_type == "log_det_sum":
                vol_loss = torch.div(torch.sum(est_T, dim=0), time_steps).slogdet().logabsdet

            identity = torch.from_numpy(np.tile(np.identity((output_dim)), (time_steps,1,1))).to(device)
            #difference = torch.sub(est_T, identity)
            had_prod = est_T * identity
            frobenius_norm = torch.mean(torch.norm(had_prod, 'fro', dim = (1,2)))

            loss = ce_loss + lam*(vol_loss) + lam_frob*frobenius_norm

            loss.backward()
            optimizer.step()
            optimizer_trans.step()

            avg_loss += loss.item()

            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        current_time = time.clock()
        scheduler.step()
        scheduler_trans.step()
        
        
        if early_stopping:
            #VALIDATION LOSS
            h = model.init_hidden(x_val.shape[0])
            model.zero_grad()
            #optimizer_trans.zero_grad()

            out, _ = model(x_val.to(device).float(), h)
            out = torch.squeeze(out, -1)

            time_steps_tensor = torch.arange(time_steps, requires_grad=False).float().unsqueeze(-1).to(device)

            est_T = trans(time_steps_tensor)

            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(est_T,clean_posterior.unsqueeze(-1)).squeeze()

            ce_loss = criterion(noisy_posterior.log().flatten(end_dim=1), y_tilde_val.flatten().to(device).long())
            #vol_loss =  torch.mean(est_T.slogdet().logabsdet)
            vol_loss = torch.mean(est_T.det())

            val_loss = ce_loss + lam*(vol_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(model, "checkpoints/"+experimentID+".pth")
                
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop

        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Avg Loss: {}, Best Val Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader), best_val_loss))
        epoch_times.append(current_time-start_time)
        
    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    #Get final T_t
    est_T = est_T.detach().cpu().numpy()

    #Transposing to get the correct Row stochastic form
    est_T = np.transpose(est_T, axes = (0,2,1))

    if early_stopping:
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)
        model.load_state_dict(torch.load("checkpoints/"+experimentID+".pth"))
        model.eval()
        model.to(device)

    return model, est_T 

#Assumes a row-stochastic matrix T_t as input
def train_RNN_forward(n_features, train_loader,  learning_rate, T_t, output_dim=2, n_layers = 1, lam = 0.01, hidden_dim=32, EPOCHS=150, verbose=True, milestones = [50,100], gamma = 0.1):

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim

    #Get total number of time steps
    time_steps = next(iter(train_loader))[0].shape[1]

    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)
    
    #Need to transpose to get column stochastic
    T_t_tranpose = np.transpose(T_t, axes = (0,2,1))
    T_t_tranpose = torch.tensor(T_t_tranpose).float().to(device)

    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    model.train()
    if verbose:
        print("Starting Training of RNN VolMin Ours model")
    epoch_times = []
    # Start training loop

    for epoch in range(1,EPOCHS+1):
        
        model.train()
        start_time = time.clock()

        avg_loss = 0.

        counter = 0
        #start_time = time.clock()

        for x, label, _, _ in train_loader:

            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, _ = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)

            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(T_t_tranpose,clean_posterior.unsqueeze(-1)).squeeze()

            loss = criterion(noisy_posterior.log().flatten(end_dim=1), label.flatten().to(device).long())
            
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        current_time = time.clock()
        scheduler.step()
        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
            
        epoch_times.append(current_time-start_time)
        
    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    return model


def train_RNN_backward(n_features, train_loader, learning_rate, T_t, output_dim=2, n_layers = 1, lam = 0.01, hidden_dim=32, EPOCHS=150, verbose=True, milestones = [50,100], gamma = 0.1):

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim

    #Get total number of time steps
    time_steps = next(iter(train_loader))[0].shape[1]

    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)
    
    #Need to transpose to get column stochastic
    #T_t_tranpose = np.transpose(T_t, axes = (0,2,1))
    #T_t_tranpose = torch.tensor(T_t_tranpose).float().to(device)
    
    T_t = torch.tensor(T_t).float().to(device)

    softmax = nn.Softmax(dim=2)
    criterion = nn.CrossEntropyLoss(reduction="none").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    model.train()
    if verbose:
        print("Starting Training of RNN VolMin Ours model")
    epoch_times = []
    # Start training loop

    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()

        avg_loss = 0.

        counter = 0
        #start_time = time.clock()
            
        for x, label, _, _ in train_loader:

            #reshaping T_t to the batch size
            batch_T_t = torch.cat(x.shape[0]*[T_t])

            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, _ = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)
            
            #evaluate loss on all possible labels for each example then concatenate
            #flattened so each timestep treated independently
            all_label_loss = []
            for c in range(output_dim):
                loss = criterion(out.flatten(end_dim=1), torch.full(label.flatten().shape, c).to(device).long())
                all_label_loss.append(loss)

            all_label_loss = torch.stack(all_label_loss, dim=-1).unsqueeze(-1)
            
            #compute backward corrected loss, then take the mean across batch x time steps
            #backward_loss = torch.bmm(batch_T_t.inverse(), all_label_loss.unsqueeze(-1)).squeeze().mean()
            backward_loss = torch.bmm(batch_T_t.inverse(), all_label_loss)[range(label.flatten().long().size(0)), label.flatten().long()].mean()
       
            backward_loss.backward()
            optimizer.step()
            

            avg_loss += backward_loss.item()

            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        current_time = time.clock()
        scheduler.step()

        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
            
        epoch_times.append(current_time-start_time)
        
    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    return model

def train_RNN_CE(n_features, train_loader, learning_rate, output_dim=2, n_layers = 1, lam = 0.01, hidden_dim=32, EPOCHS=150, verbose=True):

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim

    #Get total number of time steps
    time_steps = next(iter(train_loader))[0].shape[1]

    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)

    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    if verbose:
        print("Starting Training of RNN VolMin Ours model")
    epoch_times = []
    # Start training loop

    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()

        avg_loss = 0.

        counter = 0
        #start_time = time.clock()
        for x, label, _, _ in train_loader:

            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()

            out, _ = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)

            posterior = softmax(out)

            loss = criterion(posterior.log().flatten(end_dim=1), label.flatten().to(device).long())
            
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            current_time = time.clock()
            #print("Elapsed: ",current_time-start_time)
        current_time = time.clock()
        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
            
        epoch_times.append(current_time-start_time)
        
    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    return model

def train_RNN_volmin_T_t_losses(n_features, train_loader, T_t, learning_rate, learning_rate_trans, output_dim=2, n_layers = 1, lam = 0.01, lam_frob = 0.01, hidden_dim=32, EPOCHS=150, opt_trans = "adam", milestones=[30,60], milestones_trans = [30, 60], gamma = 0.1, gamma_trans=0.1, vol_loss_type = "sum_log_det", init = False):

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common 
    input_dim = n_features
    output_dim = output_dim
    n_layers = n_layers
    hidden_dim = hidden_dim

    #Get total number of time steps
    time_steps = next(iter(train_loader))[0].shape[1]

    # Instantiating the model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=device)

    model.to(device)

    #Initialize T generating network
    trans = T_t_gen(device, output_dim)
    if init:
        trans.apply(initialize_weights_and_biases)

    trans.to(device)
    
    trans.train()
    
    if opt_trans == "adam":
        optimizer_trans = torch.optim.Adam(trans.parameters(), lr=learning_rate_trans)
    else:
        optimizer_trans = torch.optim.SGD(trans.parameters(), lr=learning_rate_trans, momentum=0.9, weight_decay = 0.001)

    softmax = nn.Softmax(dim=2)
    criterion = nn.NLLLoss(reduction="mean").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scheduler_trans = MultiStepLR(optimizer_trans, milestones=milestones_trans, gamma=gamma_trans)

    model.train()

    epoch_times = []
    # Start training loop
    
    losses = []
    loss_type = []
    iterations = []
    iteration = 0
    T_list = []
    for epoch in range(1,EPOCHS+1):
        model.train()
        start_time = time.clock()

        avg_loss = 0.

        counter = 0

        for x, label, _, _ in train_loader:

            h = model.init_hidden(x.shape[0])
            counter += 1
            model.zero_grad()
            optimizer_trans.zero_grad()

            out, _ = model(x.to(device).float(), h)
            out = torch.squeeze(out, -1)

            time_steps_tensor = torch.arange(time_steps, requires_grad=False).float().unsqueeze(-1).to(device)

            est_T = trans(time_steps_tensor)

            clean_posterior = softmax(out)
            noisy_posterior = torch.matmul(est_T,clean_posterior.unsqueeze(-1)).squeeze()

            ce_loss = criterion(noisy_posterior.log().flatten(end_dim=1), label.flatten().to(device).long())
            if vol_loss_type == "sum_det":
                vol_loss = torch.mean(est_T.det())/time_steps
            elif vol_loss_type == "sum_log_det":
                vol_loss = torch.mean(est_T.slogdet().logabsdet)/time_steps
            elif vol_loss_type == "det_sum":
                vol_loss = torch.div(torch.sum(est_T, dim=0),time_steps).det()
            elif vol_loss_type == "log_det_sum":
                vol_loss = torch.div(torch.sum(est_T, dim=0),time_steps).slogdet().logabsdet
            elif vol_loss_type == "frob":
                norms = torch.norm(est_T, 'fro', dim=(1, 2))
                vol_loss = torch.div(torch.sum(norms, dim=0),time_steps)

            identity = torch.from_numpy(np.tile(np.identity((output_dim)), (time_steps,1,1))).to(device)
            #difference = torch.sub(est_T, identity)
            had_prod = est_T * identity
            frobenius_norm = torch.mean(torch.norm(had_prod, 'fro', dim = (1,2)))

            #print(lam_frob)
            #print(ce_loss.item(), vol_loss.item(), frobenius_norm.item())
            loss = ce_loss + lam*(vol_loss) + lam_frob*frobenius_norm


            loss.backward()
            optimizer.step()
            optimizer_trans.step()
    
            avg_loss += loss.item()

            current_time = time.clock()
            
            losses.append(ce_loss.item())
            loss_type.append("CE_loss")
            iterations.append(iteration)
            
            losses.append(vol_loss.item())
            loss_type.append("est_vol_loss")
            iterations.append(iteration)
            
            #T_t = torch.from_numpy(T_t)

            if vol_loss_type == "sum_det":
                true_vol_loss = torch.mean(torch.from_numpy(T_t).det())/time_steps
            elif vol_loss_type == "sum_log_det":
                true_vol_loss =  torch.mean(torch.from_numpy(T_t).slogdet().logabsdet)/time_steps
            elif vol_loss_type == "det_sum":
                true_vol_loss = torch.div(torch.sum(torch.from_numpy(T_t), dim=0),time_steps).det()
            elif vol_loss_type == "log_det_sum":
                true_vol_loss = torch.div(torch.sum(torch.from_numpy(T_t), dim=0),time_steps).slogdet().logabsdet
            elif vol_loss_type == "frob":
                norms = torch.norm(torch.from_numpy(T_t), 'fro', dim=(1, 2))
                true_vol_loss = torch.div(torch.sum(norms, dim=0),time_steps)

            loss_type.append("true_vol_loss")
            losses.append(true_vol_loss.item())
            iterations.append(iteration)
            
            
            iteration+=1
            
            est_T = est_T.detach().cpu().numpy()
            est_T = np.transpose(est_T, axes = (0,2,1))
            T_list.append(est_T)
            
            #print("Elapsed: ",current_time-start_time)
        current_time = time.clock()
        scheduler.step()
        scheduler_trans.step()

            
        epoch_times.append(current_time-start_time)

    #Get final T_t
    #est_T = est_T.detach().cpu().numpy()

    #Transposing to get the correct Row stochastic form
    #est_T = np.transpose(est_T, axes = (0,2,1))

    return model, est_T, losses, loss_type, iterations, T_list