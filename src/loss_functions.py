import torch
import torch.nn as nn

class backward_method(nn.Module):
    def __init__(self):
        super(backward_method, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction = "none")

    def forward(self, predictions, target, flip_probability):
        
        predictions = predictions.flatten()
        target = target.flatten()
        flipped_targets = torch.logical_not(target).float()

        loss_first = self.loss(predictions, target) #l(t,y)
        loss_second = self.loss(predictions, flipped_targets) #l(t,-y)
        
        idx = (target == 1)# indexes of samples w/ P label
        loss = loss_first.clone()  # To store l-hat
        loss[idx] = (1 - flip_probability) * loss_first[idx] - flip_probability * loss_second[idx]  # Modified loss for P samples
        loss[~idx] = (1 - flip_probability) * loss_first[~idx] - flip_probability * loss_second[~idx]  # Modified loss for N samples
        unbiased_loss = loss / (1-2*flip_probability)
        
        #print(loss_first, loss_second)
        #unbiased_loss = total_sum/total_N
        return unbiased_loss.mean()

class noise_regularized_loss(nn.Module):
    def __init__(self):
        super(noise_regularized_loss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction = "none")

    def forward(self, predictions, target, probabilities, lam=1.0):
        
        predictions = predictions.flatten()
        target = target.flatten()
        probabilities = probabilities.flatten()
        
        BCE = self.loss(predictions, target)
        regularization_term = (1-probabilities)*BCE
        
        return ((lam*regularization_term)).mean()

# class forward_method(nn.Module):
#     def __init__(self):
#         super(forward_method, self).__init__()
#         self.loss = nn.BCELoss(reduction="none")
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, predictions, target, probabilities):
        
#         predictions =self.sigmoid(predictions.flatten())
#         target = target.flatten()
#         probabilities = probabilities.flatten()
        
#         BCE = self.loss((1-probabilities)*predictions, target)
#         #regularization_term = (1-probabilities)*BCE
        
#         return BCE.mean()

class forward_method_time(nn.Module):
    def __init__(self):
        super(forward_method_time, self).__init__()
        #self.loss = nn.BCELoss(reduction="none")
        self.sigmoid = nn.Sigmoid()
    def forward(self, predictions, target, probabilities):
        
        noisy_posterior =self.sigmoid(predictions.flatten()) #p(y_tilde = 1|x)
        target = target.flatten()
        probabilities = probabilities.flatten()

        #loss_first = self.loss(noisy_posterior, target) #l(t,y)
        
        idx = (target == 1)# indexes of class 1 targets, class 0 are ~idx
        posterior = noisy_posterior.clone()  # To store l-hat
        
        #negative labels
        posterior[~idx] = (1-probabilities[~idx]) * (1-noisy_posterior[~idx]) + (probabilities[~idx]) * noisy_posterior[~idx]  # Modified posterior for N samples
        
        #positive labels
        posterior[idx] = (probabilities[idx]) * (1-noisy_posterior[idx]) + (1-probabilities[idx]) * noisy_posterior[idx]  # Modified posterior for P samples
        
        #BCE = self.loss(posterior, target)
        
        return -torch.log(posterior).mean()

class forward_method(nn.Module):
    def __init__(self):
        super(forward_method, self).__init__()
        #self.loss = nn.BCELoss(reduction="none")
        self.sigmoid = nn.Sigmoid()
    def forward(self, predictions, target, flip_probability):
        
        noisy_posterior =self.sigmoid(predictions.flatten()) #p(y_tilde = 1|x)
        target = target.flatten()

        #loss_first = self.loss(noisy_posterior, target) #l(t,y)
        
        idx = (target == 1)# indexes of class 1 targets, class 0 are ~idx
        posterior = noisy_posterior.clone()  # To store l-hat
        
        #negative labels
        posterior[~idx] = (1-flip_probability) * (1-noisy_posterior[~idx]) + (flip_probability) * noisy_posterior[~idx]  # Modified posterior for N samples
        
        #positive labels
        posterior[idx] = (flip_probability) * (1-noisy_posterior[idx]) + (1-flip_probability) * noisy_posterior[idx]  # Modified posterior for P samples
        
        #BCE = self.loss(posterior, target)
        
        return -torch.log(posterior).mean()

class backward_method_time(nn.Module):
    def __init__(self):
        super(backward_method_time, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction = "none")

    def forward(self, predictions, target, probabilities):
        
        predictions = predictions.flatten()
        target = target.flatten()
        probabilities = probabilities.flatten()
        
        flipped_targets = torch.logical_not(target).float()

        loss_first = self.loss(predictions, target) #l(t,y)
        loss_second = self.loss(predictions, flipped_targets) #l(t,-y)
        
        idx = (target == 1)# indexes of samples w/ P label
        loss = loss_first.clone()  # To store l-hat
        loss[idx] = (1 - probabilities[idx]) * loss_first[idx] - probabilities[idx] * loss_second[idx]  # Modified loss for P samples
        loss[~idx] = (1 - probabilities[~idx]) * loss_first[~idx] - probabilities[~idx] * loss_second[~idx]  # Modified loss for N samples
        unbiased_loss = loss / (1-2*probabilities)
    
        #print(loss_first, loss_second)
        #unbiased_loss = total_sum/total_N
        return unbiased_loss.mean()


#Forward loss to take in a transition matrix and works for multi-class
class forward_matrix(nn.Module):
    def __init__(self):
        super(forward_matrix, self).__init__()
        
        self.loss = nn.NLLLoss(reduction="mean")
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, predictions, target, T, num_classes):
    
        predictions = predictions.unsqueeze(2)
        
        noisy_posterior = self.logsoftmax(predictions)

        prod = torch.matmul(T, noisy_posterior).squeeze() 

        return self.loss(prod, target)
