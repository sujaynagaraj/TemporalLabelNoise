from src.data_gen import *
from src.noise import *
from src.loss_functions import *

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import time
from tqdm.notebook import tqdm
import pandas as pd
import seaborn as sns

class LogisticRegression(torch.nn.Module): # type: ignore
     def __init__(self, input_dim, output_dim, device):
         super(LogisticRegression, self).__init__()
         self.linear = nn.Linear(input_dim, output_dim)
         self.device = device
     def forward(self, x):
         outputs = (self.linear(x))
         return outputs

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

#Turn time-series dataset into iid samples at each time step for non-time series models
def make_iid(X, Y, Y_tilde, mask, probabilities):
    X_shape = X.shape
    Y_shape = Y.shape
    Y_tilde_shape = Y_tilde.shape
    mask_shape = mask.shape
    probabilities_shape = probabilities.shape
    return X.reshape(X_shape[0]*X_shape[1], X_shape[2]), Y.reshape(Y_shape[0]*Y_shape[1]), Y_tilde.reshape(Y_tilde_shape[0]*Y_tilde_shape[1]), mask.reshape(mask_shape[0]*mask_shape[1]), probabilities.reshape(probabilities_shape[0]*probabilities_shape[1])


def train_logistic_regression(n_features, train_loader, loss_function, learning_rate, output_dim, EPOCHS=5, batch_size = 32, lam=0.01, noise_probability=None, verbose = True,  keep_metrics=False):

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available() # type: ignore

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting common hyperparameters
    input_dim = n_features
    output_dim = output_dim
    # Instantiating the model
    model = LogisticRegression(input_dim, output_dim, device=device)

    #Initialize weights
    nn.init.normal_(model.linear.weight)
    nn.init.normal_(model.linear.bias)

    model.to(device)

    #Initialize T generating weights
    trans = sig_t(device, output_dim)
    trans.to(device)

    t = trans()
    trans.train()

    # Defining loss function and optimizer
    if loss_function == "natarajan_unbiased_loss":
        criterion = natarajan_unbiased_loss().to(device)
    elif loss_function == "noise_regularized_loss":
        criterion = noise_regularized_loss().to(device)
    elif loss_function == "natarajan_unbiased_loss_time":
        criterion = natarajan_unbiased_loss_time().to(device)
    elif loss_function == "forward_matrix":
        softmax = nn.Softmax(dim=1)
        criterion = nn.NLLLoss(reduction="mean").to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # type: ignore
    optimizer_trans = torch.optim.Adam(trans.parameters(), lr=learning_rate)

    model.train()

    if verbose:
        print("Starting Training of {} model".format("LR"))
    epoch_times = []
    losses = []
    losses_clean = []
    losses_noisy = []
    
    fractions_correct = []
    fractions_incorrect = []
    
    fractions_correct_noisy = []
    fractions_incorrect_noisy = []
    fractions_memorized = []

    # Start training loop
    for epoch in range(1,EPOCHS+1):
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
        for x, label, truth, mask, probabilities in train_loader:
            counter += 1
            model.zero_grad()

            if loss_function == "forward_matrix":
                optimizer_trans.zero_grad()
                t = trans()
            

            out = model(x.to(device).float())
            out = torch.squeeze(out, -1)

            if loss_function == "natarajan_unbiased_loss":
                #Batch Loss
                loss = criterion(out, label.to(device).float(), torch.as_tensor(noise_probability).to(device).float())
                if keep_metrics:
                    out = torch.flatten(out)
                    label = torch.flatten(label)
                    loss_clean = criterion(out[mask==0], label[mask==0].to(device).float(), torch.as_tensor(noise_probability).to(device).float())
                    loss_noisy = criterion(out[mask==1], label[mask==1].to(device).float(), torch.as_tensor(noise_probability).to(device).float())
            elif loss_function == "noise_regularized_loss" or loss_function == "natarajan_unbiased_loss_time":

                loss = criterion(out, label.to(device).float(), probabilities.to(device).float())
                if keep_metrics:
                    out = torch.flatten(out)
                    label = torch.flatten(label)
                    probs = torch.flatten(probabilities)
                    loss_clean = criterion(out[mask==0], label[mask==0].to(device).float(), probs[mask==0].to(device).float())
                    loss_noisy = criterion(out[mask==1], label[mask==1].to(device).float(), probs[mask==1].to(device).float())
            elif loss_function == "forward_matrix":
                clean_posterior = softmax(out)
                noisy_posterior = torch.matmul(t,clean_posterior.unsqueeze(2))

                ce_loss = criterion(noisy_posterior.squeeze().log(), label.to(device).long())

                #vol_loss = t.slogdet().logabsdet
                vol_loss = t.det()
                loss = ce_loss + lam*vol_loss

            else: #BCE Loss
                #Batch Loss
                loss = criterion(out, label.to(device).float())
                
                if keep_metrics:
                    out = torch.flatten(out)
                    label = torch.flatten(label)
                    loss_clean = criterion(out[mask==0], label[mask==0].to(device).float())
                    loss_noisy = criterion(out[mask==1], label[mask==1].to(device).float())
            predictions = torch.round(torch.sigmoid(torch.flatten(out.cpu())))
        
    
            
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

            if loss_function == "forward_matrix":
                optimizer_trans.step()

            if keep_metrics:
                    #Clean Labels
                if len(predictions[mask==0])!=0 and len(predictions[mask==1])!=0:
                    fraction_correct = len(predictions[mask==0][predictions[mask==0] == truth[mask==0]])/len(predictions[mask==0])
                    fraction_incorrect = len(predictions[mask==0][predictions[mask==0] != truth[mask==0]])/len(predictions[mask==0])

                    #Noisy Labels
                    fraction_correct_noisy = len(predictions[mask==1][predictions[mask==1] == truth[mask==1]])/len(predictions[mask==1])
                    fraction_incorrect_noisy = len(predictions[mask==1][predictions[mask==1] != truth[mask==1]])/len(predictions[mask==1])
                    fraction_memorized = len(predictions[mask==1][predictions[mask==1] == label[mask==1]])/len(predictions[mask==1])

                avg_loss_clean += loss_clean.item()
                avg_loss_noisy += loss_noisy.item()

                avg_fractions_correct += fraction_correct
                avg_fractions_incorrect += fraction_incorrect

                if len(predictions[mask==1])!=0.0:
                    avg_fractions_memorized +=fraction_memorized
                    avg_fractions_correct_noisy += fraction_correct_noisy
                    avg_fractions_incorrect_noisy += fraction_incorrect_noisy
            
        current_time = time.clock()
        if verbose and epoch%10 == 0:
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
        #print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        losses.append(avg_loss/len(train_loader))
        losses_clean.append(avg_loss_clean/len(train_loader))
        losses_noisy.append(avg_loss_noisy/len(train_loader))
        
        fractions_correct.append(avg_fractions_correct/len(train_loader))
        fractions_incorrect.append(avg_fractions_incorrect/len(train_loader))
        
        fractions_correct_noisy.append(avg_fractions_correct_noisy/len(train_loader))
        fractions_incorrect_noisy.append(avg_fractions_incorrect_noisy/len(train_loader))
        fractions_memorized.append(avg_fractions_memorized/len(train_loader))
        
        epoch_times.append(current_time-start_time)
    
    if verbose:
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    
    return (model, t), losses, losses_clean, losses_noisy, fractions_correct, fractions_incorrect, fractions_correct_noisy, fractions_incorrect_noisy, fractions_memorized

def evaluate_logistic_regression(model, x_test, y_test, output_dim = 2):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    softmax = nn.Softmax(dim=1)

    model.eval()
    model.to(device)
    outputs = []
    targets = []
    
    x_test_tensor = torch.from_numpy(x_test)
    
    out = model(x_test_tensor.to(device).float())
    out = out.squeeze(-1)

    if output_dim == 1:
        accuracy = accuracy_score(torch.round(torch.sigmoid(out)).cpu().detach(), torch.from_numpy(y_test))
    else:
        accuracy = accuracy_score(torch.round(softmax(out)).argmax(dim=1).cpu().detach(), torch.from_numpy(y_test))

    return accuracy

   