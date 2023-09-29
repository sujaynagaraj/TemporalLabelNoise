import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.autograd import Variable
import torch
import scipy
import pandas as pd
import os
import time
import sys
import pickle

def plot_results(x, y, mask, losses, accuracies, text):
    
    # creating grid for subplots
    fig = plt.figure()
    fig.set_figheight(13)
    fig.set_figwidth(20)

    ax1 = plt.subplot2grid(shape=(4, 3), loc=(0, 0), colspan=3)
    ax1.set(ylabel='X', yticklabels=[], xticklabels=[])
    ax2 = plt.subplot2grid(shape=(4, 3), loc=(1, 0), colspan=3)
    ax2.set(ylabel='Y', yticklabels=[], xticklabels=[])
    ax3 = plt.subplot2grid(shape=(4, 3), loc=(2, 0), colspan=3)
    ax3.set(ylabel='Flip Mask', yticklabels=[], xticklabels=[])
    ax4 = plt.subplot2grid((4, 3), (3, 0))
    ax4.set(ylabel='Loss over epoch')
    ax5 = plt.subplot2grid((4, 3), (3, 1), colspan=1)
    ax5.set(ylabel='Accuracies over test set')
    
    ax1.plot(x)
    ax2.plot(y)
    ax3.plot(mask)
    ax4.plot(losses)
    sns.histplot(accuracies, ax=ax5)
    fig.suptitle(text)
    plt.tight_layout()
    
    return fig


#works with only 2 dimensional data and models rn
def plot_decision_boundary(dataset, labels, model, steps=1000, color_map='Paired', title="No title"):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available() # type: ignore

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    color_map = plt.get_cmap(color_map) # type: ignore
    # Define region of interest by data limits
    amin,amax = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    bmin, bmax = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    
    a_span = np.linspace(amin, amax, steps)
    b_span = np.linspace(bmin, bmax, steps)
    
    xx, yy = np.meshgrid(a_span, b_span)

    # Make predictions across region of interest
    model.eval()
    model.to(device)
    labels_predicted = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).to(device).float()))

    # Plot decision boundary in region of interest
    labels_predicted = [0 if torch.sigmoid(value) <= 0.5 else 1 for value in labels_predicted.cpu().detach()]
    z = np.array(labels_predicted).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(xx, yy, z, cmap=color_map, alpha=0.5)

    # Get predicted labels on training data and plot
    dataset = torch.from_numpy(dataset)

    train_labels_predicted = model(dataset.to(device).float())
    train_labels_predicted = train_labels_predicted.squeeze(-1)
    ax.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap=color_map, lw=0)
    ax.set_title(title)

    plt.show()


def plot_memorization(df_losses, df_clean, df_noisy, loss):
    plt.figure()
    
    fig, ax = plt.subplots(3, sharex=False, figsize=(6,12))
    
    sns.lineplot(data = df_losses, y="Losses",x="Iterations", hue = "Labels", palette="Accent", ax = ax[0]).set(title='Loss vs Epoch')
    sns.lineplot(data = df_clean, y="Fraction",x="Iterations",hue = "Labels", palette="Accent", ax = ax[1]).set(title='Fraction Correct vs Epoch (Clean)', ylim = (0,1.0))
    sns.lineplot(data = df_noisy, y="Fraction",x="Iterations",hue = "Labels", palette="Accent", ax = ax[2]).set(title='Fraction Correct vs Epoch (Noisy)', ylim = (0,1.0))
    fig.suptitle(loss)
    plt.tight_layout()
    return fig


def plot_T_t(T_t):
    n_classes = T_t.shape[1]
    time_steps = T_t.shape[0]
    
    plt.figure()
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                sns.lineplot(y = T_t[:,i,j], x=range(time_steps), palette="Accent", label = "rho_{}_{}".format(i,j)).set(ylim = (0,0.5))
    plt.show()

def plot_T_t_unbound(T_t):
    n_classes = T_t.shape[1]
    time_steps = T_t.shape[0]
    
    plt.figure()
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                sns.lineplot(y = T_t[:,i,j], x=range(time_steps), palette="Accent", label = "rho_{}_{}".format(i,j)).set(ylim = (0,1.0))
    plt.show()

def plot_T_t_helper_unbound(T_t, ax):
    n_classes = T_t.shape[1]
    time_steps = T_t.shape[0]
    
    #ax = plt.figure(figsize=(12,8))
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                sns.lineplot(y = T_t[:,i,j], x=range(time_steps), palette="Accent", ax = ax).set(ylim = (0,1.0))

def plot_estimates(diff_df, T_t, estimated):
    plt.figure()
    
    fig, ax = plt.subplots(1,4, sharex=False, figsize=(30,5))
    
    last = estimated[-1,:,:,:]
    
    best_idx = diff_df[(diff_df['Difference']==( diff_df['Difference']).abs().min())]['Iterations'].values[0]
    best = estimated[best_idx]
    
    sns.lineplot(data = diff_df, y="Difference",x="Iterations", palette="Accent", ax = ax[0]).set(title='Difference from expected true noise', ylim = (-0.5,0.5))
    plot_T_t_helper(T_t, ax = ax[1])
    plot_T_t_helper(last, ax = ax[2])
    plot_T_t_helper(best, ax = ax[3])
    
   
    #fig.suptitle(loss)
    plt.tight_layout()
    return fig

def plot_comparison_T_t(true_T_t, est_T_t):
    n_classes = true_T_t.shape[1]
    time_steps = true_T_t.shape[0]
    
    df = {"time":[], "value":[], "flip_probability":[], "marker":[]}
    
    for t in range(time_steps):
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:

                    df["time"].append(t)
                    df["flip_probability"].append("rho_"+str(i)+str(j))
                    df["value"].append(true_T_t[t,i,j])
                    df["marker"].append("true")

                    df["time"].append(t)
                    df["flip_probability"].append("rho_"+str(i)+str(j))
                    df["value"].append(est_T_t[t,i,j])
                    df["marker"].append("est")
                    
    df = pd.DataFrame.from_dict(df)
    fig = plt.figure()
    fig = sns.lineplot(data=df, x="time", y="value",hue="flip_probability", style="marker", err_style='band').set(ylim = (0,0.5))
    return fig
    
    
def get_est_T_t(time_steps, trans, method):

    est_T_t = []
    
    for t in range(time_steps):
        if method == "ours":
            t_hat = trans(t).detach().cpu().numpy()
        else:
            t_hat = trans[t]().detach().cpu().numpy()
        est_T_t.append(t_hat)

    est_T_t = np.stack(est_T_t)

    #Transposing to get the correct Row stochastic form
    est_T_t = np.transpose(est_T_t, axes = (0,2,1))
    
    return est_T_t

def plot_list_estimates(est_T_t_list, T_t, n_classes, time_steps):

    df = {"time":[], "value":[], "flip_probability":[], "marker":[]}

    for t in range(time_steps):
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    df["time"].append(t)
                    df["flip_probability"].append("rho_"+str(i)+str(j))
                    df["value"].append(T_t[t,i,j])
                    df["marker"].append("true")
                    for est_T_t in est_T_t_list:
                        df["time"].append(t)
                        df["flip_probability"].append("rho_"+str(i)+str(j))
                        df["value"].append(est_T_t[t,i,j])
                        df["marker"].append("est")
                    
    df = pd.DataFrame.from_dict(df)
    fig = plt.figure()
    fig = sns.lineplot(data=df, x="time", y="value",hue="flip_probability", style="marker", err_style='band').set(ylim = (0,0.5))
    return fig

def plot_list_estimates_animate(est_T_t_list, T_t, n_classes, time_steps):

    df = {"time":[], "value":[], "flip_probability":[], "marker":[]}

    for t in range(time_steps):
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    df["time"].append(t)
                    df["flip_probability"].append("rho_"+str(i)+str(j))
                    df["value"].append(T_t[t,i,j])
                    df["marker"].append("true")
                    for est_T_t in est_T_t_list:
                        df["time"].append(t)
                        df["flip_probability"].append("rho_"+str(i)+str(j))
                        df["value"].append(est_T_t[t,i,j])
                        df["marker"].append("est")
                    
    df = pd.DataFrame.from_dict(df)
    #plt.figure()
    sns.lineplot(data=df, x="time", y="value",hue="flip_probability", style="marker", err_style='band').set(ylim = (0,0.5))
    plt.legend([], [], frameon=False)


