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

sys.path.insert(0,'..')

from src.data_gen import *
from src.noise import *
from src.gru import *
from src.logistic_regression import *
from src.loss_functions import *
from src.plotting import *
from data.generate_real_data import *
from data.generate_data import *


def generate_path_plot(dataset, experiment, noise_type, time_dependency = "none", model_type = "GRU", variant = "class_independent", n_states = 2, n_dims = 50, lam_frob = None):
    
    if lam_frob == None:
        if dataset == "HAR":
            lam_frob = 0.05
        elif dataset == "HAR70":
            lam_frob = 0.01
        elif dataset == "EEG_EYE":
            lam_frob = 0.001
        elif dataset == "EEG_SLEEP":
            lam_frob = 0.01
        elif dataset == "synthetic":
            lam_frob = 0.05

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    if dataset == "synthetic":
        fancy_string = f"high_var_{variant}_n_states_{n_states}_n_dims_{n_dims}_{lam_frob}"
        df_path = os.path.join(parent_dir, "results","dataframes", experiment, "synthetic", noise_type, time_dependency, model_type, "df_"+fancy_string+".csv")
  
    else:
        fancy_string = f"{dataset}_{variant}_n_states_{n_states}_{lam_frob}"
        df_path = os.path.join(parent_dir, "results","dataframes", experiment, "real", noise_type, model_type, "df_"+fancy_string+".csv")
    
    return df_path


def plot_main(dataset, limited = "time", subset_noise = False, lam_frob = None, variant = "class_independent", n_states = 2):
    limited_list = ["BCE", "Anchor", "Vol_Min", "Ours"]
    limited_list_time = ["BCE", "Anchor_Time", "Vol_Min_Time", "Ours"]
    #time_dict = {"BCE":"Time Independent", "Anchor": "Time Independent", "Anchor_Time": "Time Dependent", "Vol_Min":"Time Independent", "Vol_Min_Time":"Time Dependent", "Ours":"Time Dependent"}
    dash_dict = {"BCE":"dash", "Anchor": "dash", "Anchor_Time": "dash", "Vol_Min":"dash", "Vol_Min_Time":"dash", "Ours":"solid"}
    labels_recoding = {"BCE":"Uncorrected", "Anchor": "Anchor", "Anchor_Time": "Anchor", "Vol_Min":"VolMinNet", "Vol_Min_Time":"VolMinNet", "Ours":"TENOR"}
    
    add_time_dict = {"Uncorrected":"Uncorrected", "Anchor": "AnchorTime", "VolMinNet": "VolMinTime",  "TENOR":"TENOR"}
    
    if dataset == "HAR":
        min_y = 60
        max_y = 100
    elif dataset == "HAR70":
        min_y = 70
        max_y = 95
    elif dataset == "EEG_EYE":
        min_y = 55
        max_y = 80
    elif dataset == "EEG_SLEEP":
        min_y = 65
        max_y = 90
    elif dataset == "synthetic":
        min_y = 60
        max_y = 100
        
    if subset_noise:
        noise_types = ["basic", "sin", "mix"]
        fig, ax = plt.subplots(1, 3, figsize=(20,5))

        i = 0

        for noise_type in noise_types:
            try:
                df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob, variant = variant, n_states = n_states)
                df = pd.read_csv(df_path)
            except:
                continue
            if limited == "basic":
                df = df[df["Method"].isin(limited_list)] 
                palette=['#000000', '#81c369', '#69adc3', '#1832f7']
            elif limited == "time":
                df = df[df["Method"].isin(limited_list_time)]
                palette=['#000000', '#ec4936', "#e18c41", '#1832f7']

            df['Dash']= df['Method'].map(dash_dict)
            df['Method']= df['Method'].map(labels_recoding)
            df["Accuracy"] = df["Accuracy"]*100
            df["Noise Frequency"] = df["Noise Frequency"]*100


            sns.lineplot(data=df, x="Noise Frequency", y="Accuracy", err_style='band', 
                         hue_order = ["Uncorrected", "Anchor", "VolMinNet", "TENOR"],
                         style = "Dash",
                         style_order = ["solid", "dash"],
                         hue = "Method",
                         palette=palette,
                         dashes=["", (2, 2)],
                         linewidth = 5,
                         ax = ax[i]).set(xlim = (0, 40), ylim = (min_y, max_y))
            #sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error",hue="Method", style = "Time Dependency", err_style='band', ax = ax[1,i])

            i += 1


        handles, labels = ax[-1].get_legend_handles_labels()
        if limited == "time":
            labels = [add_time_dict[label] if label in add_time_dict.keys() else label for label in labels]

        lgd = fig.legend(handles[1:-3], labels[1:-3], loc='upper center',  bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=True, ncol=10, prop={'size': 20})

        noise_types_recoded = ["Time Independent", "Sinusoidal", "Mixed"]
        for a, col in zip(ax, noise_types_recoded):
            major_ticks = np.arange(min_y, max_y+1, 10)

            a.set_yticks(major_ticks)
            a.tick_params(axis='y', labelsize=20)
            a.tick_params(axis='x', labelsize=20)
            a.set_title(col, size=25)
            a.set_xlabel("\% of Noisy Labels", size=25)
            a.set_ylabel("Accuracy \%", size=25)
            a.grid(axis = "y")

        remove_legends = [c.get_legend().remove() for c in ax]

        plt.tight_layout()
        
        if limited == "time":
            fig.savefig("/results/figures/paper_figures/"+dataset+"+acc_vary_noise_time.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            fig.savefig("/results/figures/paper_figures/"+dataset+"+acc_vary_noise.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')


        
    else:
        noise_types = ["basic", "sig", "lin", "exp", "sin", "mix"]
        fig, ax = plt.subplots(3, 2, figsize=(16,16))

        i = 0

        for noise_type in noise_types[:3]:
            try:
                df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob, variant = variant, n_states = n_states)
                df = pd.read_csv(df_path)
            except:
                continue
            
            if limited == "basic":
                df = df[df["Method"].isin(limited_list)] 
                palette=['#000000', '#81c369', '#69adc3', '#1832f7']
            elif limited == "time":
                df = df[df["Method"].isin(limited_list_time)]
                palette=['#000000', '#ec4936', "#e18c41", '#1832f7']

            df['Dash']= df['Method'].map(dash_dict)
            df['Method']= df['Method'].map(labels_recoding)
            df["Accuracy"] = df["Accuracy"]*100
            df["Noise Frequency"] = df["Noise Frequency"]*100

            sns.lineplot(data=df, x="Noise Frequency", y="Accuracy", err_style='band', 
                         hue_order = ["Uncorrected", "Anchor", "VolMinNet", "TENOR"],
                         style = "Dash",
                         style_order = ["solid", "dash"],
                         hue = "Method",
                         palette=palette,
                         dashes=["", (2, 2)],
                         linewidth = 5,
                         ax = ax[i,0]).set(xlim = (0, 40), ylim = (min_y, max_y))
            #sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error",hue="Method", style = "Time Dependency", err_style='band', ax = ax[1,i])

            i += 1

        i = 0
        for noise_type in noise_types[3:]:
            try:
                df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob, variant = variant, n_states = n_states)
                df = pd.read_csv(df_path)
            except:
                continue
            
            if limited == "basic":
                df = df[df["Method"].isin(limited_list)] 
                palette=['#000000', '#81c369', '#69adc3', '#1832f7']
            elif limited == "time":
                df = df[df["Method"].isin(limited_list_time)]
                palette=['#000000', '#ec4936', "#e18c41", '#1832f7'] 

            df['Dash']= df['Method'].map(dash_dict)
            df['Method']= df['Method'].map(labels_recoding)
            df["Accuracy"] = df["Accuracy"]*100
            df["Noise Frequency"] = df["Noise Frequency"]*100

            sns.lineplot(data=df, x="Noise Frequency", y="Accuracy", err_style='band', 
                         hue_order = ["Uncorrected", "Anchor", "VolMinNet", "TENOR"],
                         style = "Dash",
                         style_order = ["solid", "dash"],
                         hue = "Method",
                         palette=palette,
                         dashes=["", (2, 2)],
                         linewidth = 5,
                         ax = ax[i,1]).set(xlim = (0, 40), ylim = (min_y, max_y))
            #sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error",hue="Method", style = "Time Dependency", err_style='band', ax = ax[1,i])

            i += 1

        handles, labels = ax[1,-1].get_legend_handles_labels()
        
        if limited == "time":
            labels = [add_time_dict[label] if label in add_time_dict.keys() else label for label in labels]
            
        lgd = fig.legend(handles[:-3], labels[:-3], loc='upper center',  bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=True, ncol=10, prop={'size': 15})

        noise_types_recoded = ["Time Independent", "Sigmoid", "Linear", "Exponential", "Sinusoidal", "Mixed"]
        for a, col in zip(ax[:,0], noise_types_recoded[:3]):
            major_ticks = np.arange(min_y, max_y+1, 10)

            a.set_yticks(major_ticks)
            a.tick_params(axis='y', labelsize=15)
            a.tick_params(axis='x', labelsize=15)
            a.set_title(col, size=20)
            a.set_xlabel("\% of Noisy Labels", size=20)
            a.set_ylabel("Accuracy \%", size=20)
            a.grid(axis = "y")

        for a, col in zip(ax[:,1], noise_types_recoded[3:]):
            major_ticks = np.arange(min_y, max_y+1, 10)

            a.set_yticks(major_ticks)
            a.tick_params(axis='y', labelsize=15)
            a.tick_params(axis='x', labelsize=15)
            a.set_title(col, size=20)
            a.set_xlabel("\% of Noisy Labels", size=20)
            a.set_ylabel("Accuracy \%", size=20)
            a.grid(axis = "y")

        remove_legends = [[c.get_legend().remove() for c in r] for r in ax]

        plt.tight_layout()
        
        if limited == "time":
            fig.savefig("/results/figures/paper_figures/"+dataset+"+acc_vary_noise_time_all.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            fig.savefig("/results/figures/paper_figures/"+dataset+"+acc_vary_noise_all.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_main_recon(dataset, limited = "time", subset_noise = False, lam_frob = None):
    limited_list = ["BCE", "Anchor", "Vol_Min", "Ours"]
    limited_list_time = ["BCE", "Anchor_Time", "Vol_Min_Time", "Ours"]
    #time_dict = {"BCE":"Time Independent", "Anchor": "Time Independent", "Anchor_Time": "Time Dependent", "Vol_Min":"Time Independent", "Vol_Min_Time":"Time Dependent", "Ours":"Time Dependent"}
    dash_dict = {"BCE":"dash", "Anchor": "dash", "Anchor_Time": "dash", "Vol_Min":"dash", "Vol_Min_Time":"dash", "Ours":"solid"}
    labels_recoding = {"BCE":"Uncorrected", "Anchor": "Anchor", "Anchor_Time": "Anchor", "Vol_Min":"VolMinNet", "Vol_Min_Time":"VolMinNet", "Ours":"TENOR"}
    
    add_time_dict = {"Uncorrected":"Uncorrected", "Anchor": "Anchor + Time", "VolMinNet": "VolMinNet + Time",  "TENOR":"TENOR"}
    
    if subset_noise:
        noise_types = ["basic", "sin", "mix"]
        fig, ax = plt.subplots(1, 3, figsize=(20,5))

        i = 0

        for noise_type in noise_types:
            df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob)
            df = pd.read_csv(df_path)
            if limited == "basic":
                df = df[df["Method"].isin(limited_list)]
                palette=['#000000', '#81c369', '#69adc3', '#1832f7']
            elif limited == "time":
                df = df[df["Method"].isin(limited_list_time)] 
                palette=['#000000', '#ec4936', "#e18c41", '#1832f7']

            df['Dash']= df['Method'].map(dash_dict)
            df['Method']= df['Method'].map(labels_recoding)
            df["Estimation Error"] = df["Estimation Error"] 
            df["Noise Frequency"] = df["Noise Frequency"]*100

            sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error", err_style='band', 
                         hue_order = ["Uncorrected", "Anchor", "VolMinNet", "TENOR"],
                         style = "Dash",
                         style_order = ["solid", "dash"],
                         hue = "Method",
                         palette = palette,
                         dashes=["", (2, 2)],
                         linewidth = 5,
                         ax = ax[i]).set(xlim = (0, 40), ylim = (0.0, 0.4))
            #sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error",hue="Method", style = "Time Dependency", err_style='band', ax = ax[1,i])

            i += 1

        handles, labels = ax[-1].get_legend_handles_labels()
        if limited == "time":
            labels = [add_time_dict[label] if label in add_time_dict.keys() else label for label in labels]

        lgd = fig.legend(handles[:-3], labels[:-3], loc='upper center',  bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=True, ncol=10, prop={'size': 20})

        noise_types_recoded = ["Time Independent", "Sinusoidal", "Mixed"]
        for a, col in zip(ax, noise_types_recoded):
            major_ticks = np.arange(0.0, 0.45, 0.1)

            a.set_yticks(major_ticks)
            a.tick_params(axis='y', labelsize=20)
            a.tick_params(axis='x', labelsize=20)
            a.set_title(col, size=25)
            a.set_xlabel("\% of Noisy Labels", size=25)
            a.set_ylabel("Estimation Error (MAE)", size=25)
            a.grid(axis = "y")

        remove_legends = [c.get_legend().remove() for c in ax]

        plt.tight_layout()
        
        if limited == "time":
            fig.savefig("/results/figures/paper_figures/"+dataset+"+recon_vary_noise_time.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            fig.savefig("/results/figures/paper_figures/"+dataset+"+recon_vary_noise.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')



        
    else:
        noise_types = ["basic", "sig", "lin", "exp", "sin", "mix"]
        fig, ax = plt.subplots(3, 2, figsize=(16,16))

        i = 0

        for noise_type in noise_types[:3]:
            df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob)
            df = pd.read_csv(df_path)
            
            if limited == "basic":
                df = df[df["Method"].isin(limited_list)] 
                palette=['#000000', '#81c369', '#69adc3', '#1832f7']
            elif limited == "time":
                df = df[df["Method"].isin(limited_list_time)]
                palette=['#000000', '#ec4936', "#e18c41", '#1832f7']

            df['Dash']= df['Method'].map(dash_dict)
            df['Method']= df['Method'].map(labels_recoding)
            df["Estimation Error"] = df["Estimation Error"] 
            df["Noise Frequency"] = df["Noise Frequency"]*100

            sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error", err_style='band', 
                         hue_order = ["Uncorrected", "Anchor", "VolMinNet", "TENOR"],
                         style = "Dash",
                         style_order = ["solid", "dash"],
                         hue = "Method",
                         palette=palette,
                         dashes=["", (2, 2)],
                         linewidth = 5,
                         ax = ax[i,0]).set(xlim = (0, 40), ylim = (0.0, 0.4))
            #sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error",hue="Method", style = "Time Dependency", err_style='band', ax = ax[1,i])

            i += 1

        i = 0
        for noise_type in noise_types[3:]:
            df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob)
            df = pd.read_csv(df_path)
            
            if limited == "basic":
                df = df[df["Method"].isin(limited_list)]
                palette=['#000000', '#81c369', '#69adc3', '#1832f7']
            elif limited == "time":
                df = df[df["Method"].isin(limited_list_time)]
                palette=['#000000', '#ec4936', "#e18c41", '#1832f7']

            df['Dash']= df['Method'].map(dash_dict)
            df['Method']= df['Method'].map(labels_recoding)
            df["Estimation Error"]  = df["Estimation Error"] 
            df["Noise Frequency"] = df["Noise Frequency"]*100

            sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error", err_style='band', 
                         hue_order = ["Uncorrected", "Anchor", "VolMinNet", "TENOR"],
                         style = "Dash",
                         style_order = ["solid", "dash"],
                         hue = "Method",
                         palette=palette,
                         dashes=["", (2, 2)],
                         linewidth = 5,
                         ax = ax[i,1]).set(xlim = (0, 40), ylim = (0.0, 0.4))
            #sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error",hue="Method", style = "Time Dependency", err_style='band', ax = ax[1,i])

            i += 1

        handles, labels = ax[1,-1].get_legend_handles_labels()
        
        if limited == "time":
            labels = [add_time_dict[label] if label in add_time_dict.keys() else label for label in labels]
            
        lgd = fig.legend(handles[:-3], labels[:-3], loc='upper center',  bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=True, ncol=10, prop={'size': 20})

        noise_types_recoded = ["Time Independent", "Sigmoid", "Linear", "Exponential", "Sinusoidal", "Mixed"]
        for a, col in zip(ax[:,0], noise_types_recoded[:3]):
            major_ticks = np.arange(0.0, 0.45, 0.1)

            a.set_yticks(major_ticks)
            a.tick_params(axis='y', labelsize=20)
            a.tick_params(axis='x', labelsize=20)
            a.set_title(col, size=20)
            a.set_xlabel("\% of Noisy Labels", size=25)
            a.set_ylabel("Estimation Error (MAE)", size=25)
            a.grid(axis = "y")

        for a, col in zip(ax[:,1], noise_types_recoded[3:]):
            major_ticks = np.arange(0.0, 0.45, 0.1)

            a.set_yticks(major_ticks)
            a.tick_params(axis='y', labelsize=20)
            a.tick_params(axis='x', labelsize=20)
            a.set_title(col, size=20)
            a.set_xlabel("\% of Noisy Labels", size=25)
            a.set_ylabel("Estimation Error (MAE)", size=25)
            a.grid(axis = "y")

        remove_legends = [[c.get_legend().remove() for c in r] for r in ax]

        plt.tight_layout()
        if limited == "time":
            fig.savefig("/results/figures/paper_figures/"+dataset+"+recon_vary_noise_time_all.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            fig.savefig("/results/figures/paper_figures/"+dataset+"+recon_vary_noise_all.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_all_acc(dataset, lam_frob = None, variant = "class_independent", n_states = 2):
    
    dash_dict = {"BCE":"dash", "Anchor": "dash", "Anchor_Time": "dash", "Vol_Min":"dash", "Vol_Min_Time":"dash", "Ours":"solid"}
    labels_recoding = {"BCE":"Uncorrected", "Anchor": "Anchor", "Anchor_Time": "AnchorTime", "Vol_Min":"VolMinNet", "Vol_Min_Time":"VolMinNetTime", "Ours":"TENOR"}
    
    add_time_dict = {"Uncorrected":"Uncorrected", "Anchor": "AnchorTime", "VolMinNet": "VolMinTime",  "TENOR":"TENOR"}
    
    if dataset == "HAR":
        min_y = 60
        max_y = 100
    elif dataset == "HAR70":
        min_y = 70
        max_y = 95
    elif dataset == "EEG_EYE":
        min_y = 55
        max_y = 80
    elif dataset == "EEG_SLEEP":
        min_y = 65
        max_y = 90
    elif dataset == "synthetic":
        min_y = 60
        max_y = 100
        

    noise_types = ["basic", "sig", "lin", "exp", "sin", "mix"]
    fig, ax = plt.subplots(3, 2, figsize=(16,16))

    i = 0

    for noise_type in noise_types[:3]:
        
        df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob, variant = variant, n_states = n_states)
        df = pd.read_csv(df_path)

        palette=['#000000', '#81c369','#ec4936', '#69adc3',"#e18c41", '#1832f7']
 
        df['Dash']= df['Method'].map(dash_dict)
        df['Method']= df['Method'].map(labels_recoding)
        df["Accuracy"] = df["Accuracy"]*100
        df["Noise Frequency"] = df["Noise Frequency"]*100

        sns.lineplot(data=df, x="Noise Frequency", y="Accuracy", err_style='band', 
                     hue_order = ["Uncorrected", "Anchor", "AnchorTime", "VolMinNet","VolMinTime", "TENOR"],
                     style = "Dash",
                     style_order = ["solid", "dash"],
                     hue = "Method",
                     palette=palette,
                     dashes=["", (2, 2)],
                     linewidth = 5,
                     ax = ax[i,0]).set(xlim = (0, 40), ylim = (min_y, max_y))
        #sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error",hue="Method", style = "Time Dependency", err_style='band', ax = ax[1,i])

        i += 1

    i = 0
    for noise_type in noise_types[3:]:
        df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob, variant = variant, n_states = n_states)
        df = pd.read_csv(df_path)

        palette=['#000000', '#81c369','#ec4936', '#69adc3',"#e18c41", '#1832f7']
 
        df['Dash']= df['Method'].map(dash_dict)
        df['Method']= df['Method'].map(labels_recoding)
        df["Accuracy"] = df["Accuracy"]*100
        df["Noise Frequency"] = df["Noise Frequency"]*100

        sns.lineplot(data=df, x="Noise Frequency", y="Accuracy", err_style='band', 
                     hue_order = ["Uncorrected", "Anchor", "AnchorTime", "VolMinNet","VolMinTime", "TENOR"],
                     style = "Dash",
                     style_order = ["solid", "dash"],
                     hue = "Method",
                     palette=palette,
                     dashes=["", (2, 2)],
                     linewidth = 5,
                     ax = ax[i,1]).set(xlim = (0, 40), ylim = (min_y, max_y))
        
        i += 1

    handles, labels = ax[1,-1].get_legend_handles_labels()

    labels = [add_time_dict[label] if label in add_time_dict.keys() else label for label in labels]

    lgd = fig.legend(handles[1:-3], labels[1:-3], loc='upper center',  bbox_to_anchor=(0.5, 0),
          fancybox=True, shadow=True, ncol=10, prop={'size': 20})

    noise_types_recoded = ["Time Independent", "Sigmoid", "Linear", "Exponential", "Sinusoidal", "Mixed"]
    for a, col in zip(ax[:,0], noise_types_recoded[:3]):
        major_ticks = np.arange(min_y, max_y+1, 10)

        a.set_yticks(major_ticks)
        a.tick_params(axis='y', labelsize=20)
        a.tick_params(axis='x', labelsize=20)
        a.set_title(col, size=25)
        a.set_xlabel("\% of Noisy Labels", size=25)
        a.set_ylabel("Accuracy \%", size=25)
        a.grid(axis = "y")

    for a, col in zip(ax[:,1], noise_types_recoded[3:]):
        major_ticks = np.arange(min_y, max_y+1, 10)

        a.set_yticks(major_ticks)
        a.tick_params(axis='y', labelsize=20)
        a.tick_params(axis='x', labelsize=20)
        a.set_title(col, size=20)
        a.set_xlabel("\% of Noisy Labels", size=25)
        a.set_ylabel("Accuracy \%", size=25)
        a.grid(axis = "y")

    remove_legends = [[c.get_legend().remove() for c in r] for r in ax]

    plt.tight_layout()

    fig.savefig("/results/figures/paper_figures/all_acc_"+dataset+".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    

def plot_all_recon(dataset,lam_frob = None, variant = "class_independent", n_states = 2):

    dash_dict = {"BCE":"dash", "Anchor": "dash", "Anchor_Time": "dash", "Vol_Min":"dash", "Vol_Min_Time":"dash", "Ours":"solid"}
    labels_recoding = {"BCE":"Uncorrected", "Anchor": "Anchor", "Anchor_Time": "AnchorTime", "Vol_Min":"VolMinNet", "Vol_Min_Time":"VolMinNetTime", "Ours":"TENOR"}
    
    add_time_dict = {"Uncorrected":"Uncorrected", "Anchor": "AnchorTime", "VolMinNet": "VolMinTime",  "TENOR":"TENOR"}
    

    min_y = 0.0
    max_y = 0.4

    noise_types = ["basic", "sig", "lin", "exp", "sin", "mix"]
    fig, ax = plt.subplots(3, 2, figsize=(16,16))

    i = 0

    for noise_type in noise_types[:3]:
        
        df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob, variant = variant, n_states = n_states)
        df = pd.read_csv(df_path)

        palette=['#000000', '#81c369','#ec4936', '#69adc3',"#e18c41", '#1832f7']
 
        df['Dash']= df['Method'].map(dash_dict)
        df['Method']= df['Method'].map(labels_recoding)
        df["Noise Frequency"] = df["Noise Frequency"]*100

        sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error", err_style='band', 
                     hue_order = ["Uncorrected", "Anchor", "AnchorTime", "VolMinNet","VolMinTime", "TENOR"],
                     style = "Dash",
                     style_order = ["solid", "dash"],
                     hue = "Method",
                     palette=palette,
                     dashes=["", (2, 2)],
                     linewidth = 5,
                     ax = ax[i,0]).set(xlim = (0, 40), ylim = (min_y, max_y))
        #sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error",hue="Method", style = "Time Dependency", err_style='band', ax = ax[1,i])

        i += 1

    i = 0
    for noise_type in noise_types[3:]:
        df_path = generate_path_plot(dataset, "T_estimation", noise_type, lam_frob = lam_frob, variant = variant, n_states = n_states)
        df = pd.read_csv(df_path)

        palette=['#000000', '#81c369','#ec4936', '#69adc3',"#e18c41", '#1832f7']
 
        df['Dash']= df['Method'].map(dash_dict)
        df['Method']= df['Method'].map(labels_recoding)
        df["Noise Frequency"] = df["Noise Frequency"]*100


        sns.lineplot(data=df, x="Noise Frequency", y="Estimation Error", err_style='band', 
                     hue_order = ["Uncorrected", "Anchor", "AnchorTime", "VolMinNet","VolMinTime", "TENOR"],
                     style = "Dash",
                     style_order = ["solid", "dash"],
                     hue = "Method",
                     palette=palette,
                     dashes=["", (2, 2)],
                     linewidth = 5,
                     ax = ax[i,1]).set(xlim = (0, 40), ylim = (min_y, max_y))
        
        i += 1

    handles, labels = ax[1,-1].get_legend_handles_labels()

    labels = [add_time_dict[label] if label in add_time_dict.keys() else label for label in labels]

    lgd = fig.legend(handles[1:-3], labels[1:-3], loc='upper center',  bbox_to_anchor=(0.5, 0),
          fancybox=True, shadow=True, ncol=10, prop={'size': 20})

    noise_types_recoded = ["Time Independent", "Sigmoid", "Linear", "Exponential", "Sinusoidal", "Mixed"]
    for a, col in zip(ax[:,0], noise_types_recoded[:3]):
        major_ticks = np.arange(min_y, max_y+0.1, 0.1)

        a.set_yticks(major_ticks)
        a.tick_params(axis='y', labelsize=20)
        a.tick_params(axis='x', labelsize=20)
        a.set_title(col, size=25)
        a.set_xlabel("\% of Noisy Labels", size=25)
        a.set_ylabel("Estimation Error (MAE)", size=20)
        a.grid(axis = "y")

    for a, col in zip(ax[:,1], noise_types_recoded[3:]):
        major_ticks = np.arange(min_y, max_y+0.1, 0.1)

        a.set_yticks(major_ticks)
        a.tick_params(axis='y', labelsize=20)
        a.tick_params(axis='x', labelsize=20)
        a.set_title(col, size=20)
        a.set_xlabel("\% of Noisy Labels", size=25)
        a.set_ylabel("Estimation Error (MAE)", size=20)
        a.grid(axis = "y")

    remove_legends = [[c.get_legend().remove() for c in r] for r in ax]

    plt.tight_layout()

    fig.savefig("/results/figures/paper_figures/all_recon_"+dataset+".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    

def plot_noise_helper(T_t, ax):
    n_classes = T_t.shape[1]
    time_steps = T_t.shape[0]
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                if i == 0:
                    c = "#a775e1"
                    marker = "o"
                else:
                    c = "#f95c4e"
                    marker = "x"
                sns.lineplot(y = T_t[:,i,j], x=range(time_steps), ax = ax, color = c,alpha = 0.5, marker = marker, markevery= 10,linewidth = 5,markersize = 10, label = r"$P(\tilde{{y}}_t = {0} \mid y_t = {1})$".format(i,j)).set(ylim = (0.25,0.5))

def plot_noise(variant="class_independent", subset = False):
    
    if subset:
        fig, ax = plt.subplots(1, 3, figsize=(20,5))

        T_t = T_t_generate("basic", 2, 100, a=0.40, b = 0.01, variant = variant)
        plot_noise_helper(T_t, ax = ax[0])

        T_t = T_t_generate("sin", 2, 100, a=0.095, b = 0.49, c=0.26,  variant = variant)
        plot_noise_helper(T_t, ax = ax[1])
        
        T_t = T_t_generate("mix", 2, 100, a = 0.49, b=0.005, mix_a=0.49, mix_b=0.1, mix_c = 0.35, variant = variant)
        plot_noise_helper(T_t, ax = ax[2])
        
        handles, labels = ax[-1].get_legend_handles_labels()

        lgd = fig.legend(handles, labels, loc='upper center',  bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=True, ncol=10, prop={'size': 20})

        noise_types_recoded = ["Time Independent",  "Sinusoidal", "Mixed"]
        for a, col in zip(ax, noise_types_recoded):

            a.set_title(col, size=20)
            a.tick_params(axis='y', labelsize=15)
            a.tick_params(axis='x', labelsize=15)
            a.set_xlabel("Time", size=20)
            a.set_ylabel("Flipping Probability", size=20)

        remove_legends = [c.get_legend().remove() for c in ax]

        plt.tight_layout()

        plt.savefig("/results/figures/paper_figures/noise_viz.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
      

        
    else:
        
        fig, ax = plt.subplots(3, 2, figsize=(16,16))

        T_t = T_t_generate("basic", 2, 100, a=0.40, b = 0.01, variant = variant)
        plot_noise_helper(T_t, ax = ax[0,0])

        T_t = T_t_generate("exp", 2, 100, a=0.49, b = 0.004, variant = variant)
        plot_noise_helper(T_t, ax = ax[0,1]) 

        T_t = T_t_generate("sig", 2, 100, a=0.49, b = 0.1, c = 0.3, variant = variant)
        plot_noise_helper(T_t, ax = ax[1,0])

        T_t = T_t_generate("sin", 2, 100, a=0.095, b = 0.49, c=0.26,  variant = variant)
        plot_noise_helper(T_t, ax = ax[1,1])

        T_t = T_t_generate("lin", 2, 100, a=0.49, b = 0.31, variant = variant)
        plot_noise_helper(T_t, ax = ax[2,0])

        T_t = T_t_generate("mix", 2, 100, a = 0.49, b=0.005, mix_a=0.49, mix_b=0.1, mix_c = 0.35, variant = "class_independent")
        plot_noise_helper(T_t, ax = ax[2,1])
        
    
        
        handles, labels = ax[1,-1].get_legend_handles_labels()

        lgd = fig.legend(handles, labels, loc='upper center',  bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=True, ncol=10, prop={'size': 15})

        noise_types_recoded = ["Time Independent", "Sigmoid", "Linear", "Exponential", "Sinusoidal", "Mixed"]
        for a, col in zip(ax[:,0], noise_types_recoded[:3]):

            a.set_title(col, size=20)
            a.tick_params(axis='y', labelsize=15)
            a.tick_params(axis='x', labelsize=15)
            a.set_xlabel("Time", size=20)
            a.set_ylabel("Flipping Probability", size=20)

        for a, col in zip(ax[:,1], noise_types_recoded[3:]):

            a.set_title(col, size=20)
            a.tick_params(axis='y', labelsize=15)
            a.tick_params(axis='x', labelsize=15)
            a.set_xlabel("Time", size=20)
            a.set_ylabel("Flipping Probability", size=20)

        remove_legends = [[c.get_legend().remove() for c in r] for r in ax]

        plt.tight_layout()
        
        plt.savefig("/results/figures/paper_figures/noise_viz_all.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
      
def plot_motivation(subset_noise = False,variant = "class_independent"):
    
    labels_recoding = {"CE":"Uncorrected", "Forward_Time_Dependent": "Forward", "Backward_Time_Dependent": "Backward", "Forward_Time_Independent": "Forward", "Backward_Time_Independent": "Backward"}
    time_dict = {"CE": r"Static", "Forward_Time_Dependent": "Temporal", "Backward_Time_Dependent": "Temporal", "Forward_Time_Independent": r"Static", "Backward_Time_Independent": r"Static"}
    
    
    if subset_noise:
        noise_types = ["basic", "sin", "mix"]
        fig, ax = plt.subplots(1, 3, figsize=(20,5))

        i = 0

        for noise_type in noise_types:
            df_path = generate_path_plot("synthetic","motivation", noise_type, variant = variant)
            df = pd.read_csv(df_path)

            df["Noise Frequency"] = df["noise_frequency"]*100
            df["Accuracy"] = df["accuracy"]*100
            df["Method"] = df["methods"]
            df['Time']= df['Method'].map(time_dict)
            df['Method']= df['Method'].map(labels_recoding)

            sns.lineplot(data=df, x="Noise Frequency", y="Accuracy", err_style='band', 
                         hue_order = ["Uncorrected", "Forward", "Backward"],
                         style = "Time",
                         style_order = ["Temporal", r"Static"],
                         hue = "Method",
                         palette=['#000000', '#81c369', '#69adc3'],
                         dashes=["", (2, 2)],
                         linewidth = 5,
                         ax = ax[i]).set(xlim = (0, 40), ylim = (60, 105))
            i += 1

       
        handles, labels = ax[-1].get_legend_handles_labels()

        labels.pop(4)
        handles.pop(4)

        labels.pop(0)
        handles.pop(0)

        lgd = fig.legend(handles, labels, loc='upper center',  bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=True, ncol=10, prop={'size': 20})

        noise_types_recoded = ["Time Independent",  "Sinusoidal", "Mixed"]

        for a, col in zip(ax, noise_types_recoded):
            major_ticks = np.arange(60, 101, 10)

            a.set_yticks(major_ticks)
            a.tick_params(axis='y', labelsize=20)
            a.tick_params(axis='x', labelsize=20)
            a.set_title(col, size=25)
            a.set_xlabel("\% of Noisy Labels", size=25)
            a.set_ylabel("Accuracy \%", size=25)
            a.grid(axis = "y")


        remove_legends = [c.get_legend().remove() for c in ax]

        plt.tight_layout()

        plt.savefig("/results/figures/paper_figures/motivation.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
       

        
    else:
        
        noise_types = ["basic", "sig", "lin", "exp", "sin", "mix"]
        fig, ax = plt.subplots(3, 2, figsize=(16,16))

        i = 0

        for noise_type in noise_types[:3]:
            df_path = generate_path_plot("synthetic","motivation", noise_type, variant = variant)
            df = pd.read_csv(df_path)

            df["Noise Frequency"] = df["noise_frequency"]*100
            df["Accuracy"] = df["accuracy"]*100
            df["Method"] = df["methods"]
            df['Time']= df['Method'].map(time_dict)
            df['Method']= df['Method'].map(labels_recoding)

            sns.lineplot(data=df, x="Noise Frequency", y="Accuracy", err_style='band', 
                         hue_order = ["Uncorrected", "Forward", "Backward"],
                         style = "Time",
                         style_order = ["Temporal", r"Static"],
                         hue = "Method",
                         palette=['#000000', '#81c369', '#69adc3'],
                         dashes=["", (2, 2)],
                         linewidth = 5,
                         ax = ax[i,0]).set(xlim = (0, 40), ylim = (60, 105))
            i += 1

        i = 0
        for noise_type in noise_types[3:]:
            df_path = generate_path_plot("synthetic","motivation", noise_type, variant = variant)
            df = pd.read_csv(df_path)

            df["Noise Frequency"] = df["noise_frequency"]*100
            df["Accuracy"] = df["accuracy"]*100
            df["Method"] = df["methods"]
            df['Time']= df['Method'].map(time_dict)
            df['Method']= df['Method'].map(labels_recoding)

            sns.lineplot(data=df, x="Noise Frequency", y="Accuracy", err_style='band', 
                         hue_order = ["Uncorrected", "Forward", "Backward"],
                         style = "Time",
                         style_order = ["Temporal", r"Static"],
                         hue = "Method",
                         palette=['#000000', '#81c369', '#69adc3'],
                         dashes=["", (2, 2)],
                         linewidth = 5,
                         ax = ax[i,1]).set(xlim = (0, 40), ylim = (60, 105))
            i += 1

        handles, labels = ax[1,-1].get_legend_handles_labels()
        #labels = [labels_recoding[label] for label in labels]

        labels.pop(4)
        handles.pop(4)

        labels.pop(0)
        handles.pop(0)

        lgd = fig.legend(handles, labels, loc='upper center',  bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=True, ncol=10, prop={'size': 15})

        noise_types_recoded = ["Time Independent", "Sigmoid", "Linear", "Exponential", "Sinusoidal", "Mixed"]

        for a, col in zip(ax[:,0], noise_types_recoded[:3]):
            major_ticks = np.arange(60, 101, 10)

            a.set_yticks(major_ticks)
            a.tick_params(axis='y', labelsize=15)
            a.tick_params(axis='x', labelsize=15)
            a.set_title(col, size=20)
            a.set_xlabel("\% of Noisy Labels", size=20)
            a.set_ylabel("Accuracy \%", size=20)
            a.grid(axis = "y")

        for a, col in zip(ax[:,1], noise_types_recoded[3:]):
            major_ticks = np.arange(60, 101, 10)

            a.set_yticks(major_ticks)
            a.tick_params(axis='y', labelsize=15)
            a.tick_params(axis='x', labelsize=15)
            a.set_title(col, size=20)
            a.set_xlabel("\% of Noisy Labels", size=20)
            a.set_ylabel("Accuracy \%", size=20)
            a.grid(axis = "y")

        remove_legends = [[c.get_legend().remove() for c in r] for r in ax]

        plt.tight_layout()
        plt.savefig("/results/figures/paper_figures/motivation_all.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
       
