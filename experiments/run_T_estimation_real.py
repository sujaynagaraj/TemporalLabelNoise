import os
import time
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0,'..')

import argparse
from random import SystemRandom

from src.data_gen import *
from src.gru import *
from src.noise import *
from src.plotting import *

from sklearn.utils import shuffle
from data.generate_real_data import *
from data.generate_data import *

def experiment_body(m, n_dims, train_loader, x_test, y_test, experimentID, learning_rate, n_states, epochs, length, x_val = None, y_tilde_val = None, lam = 0.0001, lam_frob = 0.01, opt="adam", milestones = [], milestones_trans= [50, 100, 150, 200], gamma = 0.5, gamma_trans = 0.5, opt_trans = "adam", learning_rate_trans = 0.01, vol_loss_type = "sum_log_det"):
    print(m)
    print("LR ", learning_rate)
    print("lam ", lam)
    print("LR trans ", learning_rate_trans)
    print("milestones ", milestones)
    print("milestones_trans", milestones_trans)
    print("opt ", opt)
    print("opt_trans ", opt_trans)
    print("lam_frob ", lam_frob)


    if m == "BCE":
        milestones_BCE = milestones
        model, _, _, est_T_t = train_RNN_anchor(n_dims, train_loader, experimentID, learning_rate, n_states, EPOCHS = epochs, opt=opt, milestones = milestones_BCE, gamma = gamma)
        est_T_t = np.tile(np.identity((n_states)), (length,1,1)) #BCE Has no T-Estimation ability, so returns identity matrix at each step

    elif m == "Anchor":
        warmup = 25
        milestones_BCE = milestones
        model, _, _, est_T_t = train_RNN_anchor_warmup(n_dims, train_loader,  experimentID, learning_rate, output_dim = n_states, EPOCHS = epochs, warmup=warmup, verbose=True, quantile = 0.97, milestones = milestones_BCE, gamma = gamma)

    elif m == "Anchor_Time":
        warmup = 25
        milestones_BCE = milestones
        model, _, _, est_T_t = train_RNN_anchor_time_warmup(n_dims, train_loader,  experimentID, learning_rate, output_dim = n_states, EPOCHS = epochs, warmup=warmup, verbose=True, quantile = 0.97,  milestones = milestones_BCE, gamma = gamma)

    elif m == "Vol_Min":
        model, est_T = train_RNN_volmin2(n_dims, train_loader,  experimentID, learning_rate, learning_rate_trans = learning_rate_trans, output_dim=n_states, lam = lam,  EPOCHS=epochs, verbose=True, opt_trans=opt_trans,  milestones = milestones , milestones_trans = milestones_trans, gamma = gamma, gamma_trans = gamma_trans)
        est_T_t = np.tile(est_T, (length,1,1))
        
    elif m == "Vol_Min_Time":
        model, trans_list = train_RNN_volmin_time2(n_dims, train_loader,  experimentID, learning_rate, learning_rate_trans = learning_rate_trans, output_dim=n_states, lam = lam, lam_frob= lam_frob, EPOCHS=epochs, verbose=True, opt_trans=opt_trans,  milestones = milestones, milestones_trans = milestones_trans, gamma = gamma, gamma_trans = gamma_trans, vol_loss_type = vol_loss_type)
        est_T_t = get_est_T_t(length, trans_list, "vol_min_t")
        
    elif m == "Ours":
        model, est_T_t = train_RNN_volmin_T_t(n_dims, train_loader, experimentID, learning_rate, learning_rate_trans = learning_rate_trans, output_dim=n_states, lam = lam, lam_frob=lam_frob, EPOCHS=epochs, verbose=True, opt_trans=opt_trans,  milestones = milestones, milestones_trans = milestones_trans, gamma = gamma, gamma_trans = gamma_trans, vol_loss_type = vol_loss_type)
        
    _, acc = evaluate_RNN(model, x_test, y_test)

    return model, est_T_t, acc

parser = argparse.ArgumentParser('T Estimation')

parser.add_argument('--dataset_type', type=str, default='HAR', help="Feature sets to include:")
parser.add_argument('--n_states', type =int, default=2, help="number of states/classes in data")
parser.add_argument('--length', type =int, default=100, help="length (time) of each sample")
parser.add_argument('--n_iterations', type = int, default = 15, help="number of iterations to train model on dataset")
parser.add_argument('--noise_type', type=str, default='basic', help="basic, class, or time")
parser.add_argument('--variant', type=str, default='class_independent', help="class independent or dependent noise")
parser.add_argument('--batch_size', type=int, default=32, help="specify batch size")
parser.add_argument('--epochs', type=int, default=150, help="specify num epochs")
parser.add_argument('--learning_rate', type=float, default=0.01, help="specify learning rate")
parser.add_argument('--mask_start_prob', type=float, default=0.9, help="prob of first label being clean")
parser.add_argument('--lam', type=float, default=0.0001, help="lambda tradeoff in volmin loss")
parser.add_argument('--a', type=float, default=0.49, help="a parameter for noise function")
parser.add_argument('--b', type=float, default=0.1, help="b parameter for noise function")
parser.add_argument('--center', type=int, default=20, help="timestep to center sigmoid function around")
parser.add_argument('--sig_flip', type=bool, default=False, help="to flip or not to flip sigmoid noise around center y")
parser.add_argument('--add_noise_var', type=float, default=0.1, help="amount of additive noise to add")
parser.add_argument('--model_type', type=str, default="GRU", help="Type of Model")

args = parser.parse_args()


#####################################################################################################

if __name__ == '__main__':

    experimentID = str(int(SystemRandom().random()*100000))

    start = time.time()

    print('Starting T_estimation on Real Data')
    print("Dataset: ", args.dataset_type)
    print("Noise Type: ", args.noise_type)

    dataset_type = args.dataset_type
    n_states = args.n_states
    length = args.length
    n_iterations = args.n_iterations
    noise_type = args.noise_type
    variant = args.variant
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    method = args.noise_type
    mask_start_prob = args.mask_start_prob
    lam = args.lam
    a = args.a
    b = args.b
    center = args.center
    sig_flip = args.sig_flip
    add_noise_var = args.add_noise_var
    model_type = args.model_type

    methods_list = ["BCE","Anchor", "Anchor_Time" ,"Vol_Min", "Vol_Min_Time", "Ours"]

    if dataset_type == "EEG_EYE":

        batch_size = 128
        
        epochs = 150
        opt = "adam"
        opt_trans = "adam"
        milestones = []
        milestones_trans = [50,100]
        lam = 0.0001
        lam_frob = 0.001
        learning_rate = 0.01
        learning_rate_trans = 0.01
        gamma = 0.5
        gamma_trans = 0.5
        warmup = 25

        vol_loss_type = "sum_log_det"
        
        
    elif dataset_type == "EEG_SLEEP":
        
        batch_size = 512
        
        epochs = 150
        opt = "adam"
        opt_trans = "adam"
        milestones = []
        milestones_trans = [50,100]
        lam = 0.0001
        lam_frob = 0.01
        learning_rate = 0.01
        learning_rate_trans = 0.01
        gamma = 0.5
        gamma_trans = 0.5
        warmup = 25

        vol_loss_type = "sum_log_det"
       
    elif dataset_type == "HAR":

        batch_size = 64

        epochs = 150
        opt = "adam"
        opt_trans = "adam"
        milestones = []
        milestones_trans = [50,100]
        lam = 0.0001
        lam_frob = 0.05
        learning_rate = 0.01
        learning_rate_trans = 0.01
        gamma = 0.5
        gamma_trans = 0.5
        warmup = 25

        vol_loss_type = "sum_log_det"

        
    elif dataset_type == "HAR70":

        batch_size = 256

        epochs = 150
        opt = "adam"
        opt_trans = "adam"
        milestones = []
        milestones_trans = [50,100]
        lam = 0.0001
        lam_frob = 0.01
        learning_rate = 0.01
        learning_rate_trans = 0.01
        gamma = 0.5
        gamma_trans = 0.5
        warmup = 25

        vol_loss_type = "sum_log_det"

    ###############

    ##################################################################
    filename = generate_filename_real(dataset_type, length, n_states)
    path = "/data/real/processed/"
    fancy_string = f"{args.dataset_type}_{args.variant}_n_states_{args.n_states}_{lam_frob}"

    #results path
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    if dataset_type == "HAR":
        X_clean_train, Y_clean_train = load_data_real(filename+"_train",path)
        X_clean_test, Y_clean_test = load_data_real(filename+"_test",path)
        n_dims = X_clean_train.shape[2]
    else:
        X_clean, Y_clean = load_data_real(filename,path)
        n_dims = X_clean.shape[2]

    accuracies = []
    noise_frequencies = []
    methods = []
    T_errors = []

    ##################################################################
    #Run Experiments
    if args.noise_type == "basic":
         for a in ([0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]):

            if dataset_type == "HAR":
                x_train, y_train, y_tilde_train, mask_train, T_t = add_noise(X_clean_train, 
                                                    Y_clean_train, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    variant = variant)
                x_test, y_test, y_tilde_test, mask_test, T_t = add_noise(X_clean_test, 
                                                    Y_clean_test, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    variant = variant)
                                                   
            else:
                X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                    Y_clean, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    variant = variant)

                x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)


            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

            freq = get_flip_frequency(T_t)
            print("Noise Freq: ", '{0:.2f}'.format(freq))

            for m in methods_list:
                T_estimated = []
                for iteration in (range(n_iterations)):

                    model, est_T_t, acc = experiment_body(m, n_dims, train_loader, x_test, y_test, experimentID, learning_rate, n_states, epochs, length, lam = lam,lam_frob=lam_frob, opt=opt, milestones = milestones, milestones_trans= milestones_trans, gamma = gamma, gamma_trans = gamma_trans, opt_trans = opt_trans, learning_rate_trans = learning_rate_trans, vol_loss_type = vol_loss_type) 
                    accuracies.append(acc)
                    noise_frequencies.append(float('{0:.2f}'.format(freq)))
                    methods.append(m)
                    mae = np.absolute(est_T_t - T_t).mean().item()
                    T_errors.append(mae)
                    T_estimated.append(est_T_t)

                if '{0:.2f}'.format(freq) in ['0.00','0.20', '0.40']:
                    fancy_string_freq = fancy_string + "_freq_" + str('{0:.2f}'.format(freq))
                    if not os.path.exists(os.path.join(parent_dir, "results", "figures", "reconstruction", "real", args.noise_type, m)):
                        os.makedirs(os.path.join(parent_dir, "results", "figures","reconstruction", "real",args.noise_type,  m))
        
                    image_path_recon = os.path.join(parent_dir, "results","figures", "reconstruction","real", args.noise_type, m, fancy_string_freq+".png")
                    fig = plot_list_estimates(T_estimated, T_t, n_states, length)
                    plt.savefig(image_path_recon)


    elif args.noise_type == "exp":
        for b in (reversed([0.004, 0.008, 0.01, 0.015, 0.022, 0.025, 0.03, 0.06, 1.0])):
            if b == 1.0:
                a = 0.0
            else:
                a = 0.49
            if dataset_type == "HAR":
                x_train, y_train, y_tilde_train, mask_train, T_t = add_noise(X_clean_train, 
                                                    Y_clean_train, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    variant = variant)
                x_test, y_test, y_tilde_test, mask_test, T_t = add_noise(X_clean_test, 
                                                    Y_clean_test, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    variant = variant)

            else:
                X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                    Y_clean, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    variant = variant)



            if dataset_type != "HAR":
                x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

            freq = get_flip_frequency(T_t)
            print("Noise Freq: ", '{0:.2f}'.format(freq))

            for m in methods_list:
                T_estimated = []
                for iteration in (range(n_iterations)):

                    model, est_T_t, acc = experiment_body(m, n_dims, train_loader, x_test, y_test, experimentID, learning_rate, n_states, epochs, length, lam = lam, lam_frob=lam_frob, opt=opt, milestones = milestones, milestones_trans= milestones_trans, gamma = gamma, gamma_trans = gamma_trans, opt_trans = opt_trans, learning_rate_trans = learning_rate_trans, vol_loss_type = vol_loss_type) 

                    accuracies.append(acc)
                    noise_frequencies.append(float('{0:.2f}'.format(freq)))
                    methods.append(m)
                    mae = np.absolute(est_T_t - T_t).mean().item()
                    T_errors.append(mae)
                    T_estimated.append(est_T_t)
                if '{0:.2f}'.format(freq) in ['0.00','0.20', '0.40']:
                    fancy_string_freq = fancy_string + "_freq_" + str('{0:.2f}'.format(freq))
                    if not os.path.exists(os.path.join(parent_dir, "results", "figures", "reconstruction","real", args.noise_type , m)):
                        os.makedirs(os.path.join(parent_dir, "results", "figures","reconstruction", "real", args.noise_type,  m))
        
                    image_path_recon = os.path.join(parent_dir, "results","figures", "reconstruction", "real", args.noise_type, m, fancy_string_freq+".png")
                    fig = plot_list_estimates(T_estimated, T_t, n_states, length)
                    plt.savefig(image_path_recon)
        

    elif args.noise_type == "sig":
        a_list = [0,   0.2,  0.3, 0.4,  0.45, 0.49, 0.49, 0.49]
        b_list = [0.1,  0.1,  0.1, 0.1,  0.1, 0.1, 0.1, 0.1,]
        c_list = [0, 0, 0, 0,  0.05, 0.1, 0.2, 0.3]

        for a, b, c in zip(a_list, b_list, c_list):
            if dataset_type == "HAR":
                x_train, y_train, y_tilde_train, mask_train, T_t = add_noise(X_clean_train, 
                                                    Y_clean_train, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    c = c,
                                                    variant = variant)
                x_test, y_test, y_tilde_test, mask_test, T_t = add_noise(X_clean_test, 
                                                    Y_clean_test, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    c = c,
                                                    variant = variant)
                                                   
            else:
                X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                    Y_clean, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    c = c,
                                                    variant = variant)


            if dataset_type != "HAR":
                x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

            freq = get_flip_frequency(T_t)
            print("Noise Freq: ", '{0:.2f}'.format(freq))

            for m in methods_list:
                T_estimated = []
                for iteration in (range(n_iterations)):
                    
                    model, est_T_t, acc = experiment_body(m, n_dims, train_loader, x_test, y_test, experimentID, learning_rate, n_states, epochs, length, lam = lam,lam_frob=lam_frob, opt=opt, milestones = milestones, milestones_trans= milestones_trans, gamma = gamma, gamma_trans = gamma_trans, opt_trans = opt_trans, learning_rate_trans = learning_rate_trans, vol_loss_type = vol_loss_type) 

                    accuracies.append(acc)
                    noise_frequencies.append(float('{0:.2f}'.format(freq)))
                    methods.append(m)
                    mae = np.absolute(est_T_t - T_t).mean().item()
                    T_errors.append(mae)
                    T_estimated.append(est_T_t)
                if '{0:.2f}'.format(freq) in ['0.00','0.20', '0.40']:
                    fancy_string_freq = fancy_string + "_freq_" + str('{0:.2f}'.format(freq))
                    if not os.path.exists(os.path.join(parent_dir, "results", "figures", "reconstruction", "real", args.noise_type ,m)):
                        os.makedirs(os.path.join(parent_dir, "results", "figures","reconstruction","real", args.noise_type,  m))
        
                    image_path_recon = os.path.join(parent_dir, "results","figures", "reconstruction","real",args.noise_type, m, fancy_string_freq+".png")
                    fig = plot_list_estimates(T_estimated, T_t, n_states, length)
                    plt.savefig(image_path_recon)
                


    elif args.noise_type == "sin":
        a_list = [0.095,  0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095]
        b_list = [0,  0.15, 0.2, 0.25, 0.3, 0.33, 0.41, 0.45, 0.49, 0.49, 0.49]
        c_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1,  0.26]

        for a, b, c in zip(a_list, b_list, c_list):
            if dataset_type == "HAR":
                x_train, y_train, y_tilde_train, mask_train, T_t = add_noise(X_clean_train, 
                                                    Y_clean_train, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    c = c,
                                                    variant = variant)
                x_test, y_test, y_tilde_test, mask_test, T_t = add_noise(X_clean_test, 
                                                    Y_clean_test, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    c = c,
                                                    variant = variant)


            else:
                X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                    Y_clean, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    c = c,
                                                    variant = variant)


            if dataset_type != "HAR":
                x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

            freq = get_flip_frequency(T_t)
            print("Noise Freq: ", '{0:.2f}'.format(freq))

            for m in methods_list:
                T_estimated = []
                for iteration in (range(n_iterations)):
                    
                    model, est_T_t, acc = experiment_body(m, n_dims, train_loader, x_test, y_test, experimentID, learning_rate, n_states, epochs, length, lam = lam, lam_frob=lam_frob, opt=opt, milestones = milestones, milestones_trans= milestones_trans, gamma = gamma, gamma_trans = gamma_trans, opt_trans = opt_trans, learning_rate_trans = learning_rate_trans, vol_loss_type = vol_loss_type) 

                    accuracies.append(acc)
                    noise_frequencies.append(float('{0:.2f}'.format(freq)))
                    methods.append(m)
                    mae = np.absolute(est_T_t - T_t).mean().item()
                    T_errors.append(mae)
                    T_estimated.append(est_T_t)
                if '{0:.2f}'.format(freq) in ['0.00','0.20', '0.40']:
                    fancy_string_freq = fancy_string + "_freq_" + str('{0:.2f}'.format(freq))
                    if not os.path.exists(os.path.join(parent_dir, "results", "figures", "reconstruction","real", args.noise_type, m)):
                        os.makedirs(os.path.join(parent_dir, "results", "figures","reconstruction","real", args.noise_type, m))
        
                    image_path_recon = os.path.join(parent_dir, "results","figures", "reconstruction","real",args.noise_type, m, fancy_string_freq+".png")
                    fig = plot_list_estimates(T_estimated, T_t, n_states, length)
                    plt.savefig(image_path_recon)

    elif args.noise_type == "lin":
        a_list = [0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 0.4, 0.35, 0.3, 0.25, 0.2, 0]
        b_list = [0.31, 0.3, 0.2, 0.1, 0.05, 0, 0, 0, 0, 0, 0, 0]
        for a,b in zip(reversed(a_list), reversed(b_list)):
            if a>=b:
                if dataset_type == "HAR":
                    x_train, y_train, y_tilde_train, mask_train, T_t = add_noise(X_clean_train, 
                                                    Y_clean_train, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    variant = variant)
                    x_test, y_test, y_tilde_test, mask_test, T_t = add_noise(X_clean_test, 
                                                        Y_clean_test, 
                                                        args.noise_type, 
                                                        n_states, 
                                                        a = a,
                                                        b = b,
                                                        variant = variant)

                                                

                else:
                    X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                        Y_clean, 
                                                        args.noise_type, 
                                                        n_states, 
                                                        a = a,
                                                        b = b,
                                                        variant = variant)



        
                if dataset_type != "HAR":
                    x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

                train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

                freq = get_flip_frequency(T_t)
                print("Noise Freq: ", '{0:.2f}'.format(freq))

                for m in methods_list:
                    T_estimated = []
                    for iteration in (range(n_iterations)):
                        
                        
                        model, est_T_t, acc = experiment_body(m, n_dims, train_loader, x_test, y_test, experimentID, learning_rate, n_states, epochs, length, lam = lam, lam_frob=lam_frob, opt=opt, milestones = milestones, milestones_trans= milestones_trans, gamma = gamma, gamma_trans = gamma_trans, opt_trans = opt_trans, learning_rate_trans = learning_rate_trans, vol_loss_type = vol_loss_type) 

                        accuracies.append(acc)
                        noise_frequencies.append(float('{0:.2f}'.format(freq)))
                        methods.append(m)
                        mae = np.absolute(est_T_t - T_t).mean().item()
                        T_errors.append(mae)
                        T_estimated.append(est_T_t)
                    if '{0:.2f}'.format(freq) in ['0.00','0.20', '0.40']:
                        fancy_string_freq = fancy_string + "_freq_" + str('{0:.2f}'.format(freq))
                        if not os.path.exists(os.path.join(parent_dir, "results", "figures", "reconstruction","real", args.noise_type, m)):
                            os.makedirs(os.path.join(parent_dir, "results", "figures","reconstruction","real", args.noise_type, m))
            
                        image_path_recon = os.path.join(parent_dir, "results","figures", "reconstruction","real",args.noise_type, m, fancy_string_freq+".png")
                        fig = plot_list_estimates(T_estimated, T_t, n_states, length)
                        plt.savefig(image_path_recon)

    elif args.noise_type == "mix":
        a = 0.49
        b_list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 1.0]
        mix_a_list = [0.49, 0.49, 0.49, 0.49, 0.45, 0.4, 0.35, 0.3, 0.25,  0]
        mix_c_list = [0.35, 0.3, 0.25, 0.2, 0.1, 0, 0, 0, 0, 0]
        mix_b = 0.1

        for b, mix_a, mix_c in zip(reversed(b_list), reversed(mix_a_list), reversed(mix_c_list)):
            if b == 1.0:
                a = 0.0
            else:
                a = 0.49
            if dataset_type == "HAR":
                x_train, y_train, y_tilde_train, mask_train, T_t = add_noise(X_clean_train, 
                                                    Y_clean_train, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    mix_a = mix_a,
                                                    mix_b = mix_b,
                                                    mix_c = mix_c,
                                                    variant = variant)
                x_test, y_test, y_tilde_test, mask_test, T_t = add_noise(X_clean_test, 
                                                    Y_clean_test, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    mix_a = mix_a,
                                                    mix_b = mix_b,
                                                    mix_c = mix_c,
                                                    variant = variant)
                                                   

            else:
                X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                    Y_clean, 
                                                    args.noise_type, 
                                                    n_states, 
                                                    a = a,
                                                    b = b,
                                                    mix_a = mix_a,
                                                    mix_b = mix_b,
                                                    mix_c = mix_c,
                                                    variant = variant)


            freq = get_flip_frequency(T_t)
            print("Noise Freq: ", '{0:.2f}'.format(freq))


            if dataset_type != "HAR":
                x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

            for m in methods_list:
                T_estimated = []
                for iteration in (range(n_iterations)):
                    
                    model, est_T_t, acc = experiment_body(m, n_dims, train_loader, x_test, y_test, experimentID, learning_rate, n_states, epochs, length, lam = lam, lam_frob=lam_frob, opt=opt, milestones = milestones, milestones_trans= milestones_trans, gamma = gamma, gamma_trans = gamma_trans, opt_trans = opt_trans, learning_rate_trans = learning_rate_trans, vol_loss_type = vol_loss_type) 
                    
                    accuracies.append(acc)
                    noise_frequencies.append(float('{0:.2f}'.format(freq)))
                    methods.append(m)
                    mae = np.absolute(est_T_t - T_t).mean().item()
                    T_errors.append(mae)
                    T_estimated.append(est_T_t)
                if '{0:.2f}'.format(freq) in ['0.00','0.20', '0.40']:
                    fancy_string_freq = fancy_string + "_freq_" + str('{0:.2f}'.format(freq))
                    if not os.path.exists(os.path.join(parent_dir, "results", "figures", "reconstruction","real", args.noise_type, m)):
                        os.makedirs(os.path.join(parent_dir, "results", "figures","reconstruction","real", args.noise_type, m))
        
                    image_path_recon = os.path.join(parent_dir, "results","figures", "reconstruction","real",args.noise_type, m, fancy_string_freq+".png")
                    fig = plot_list_estimates(T_estimated, T_t, n_states, length)
                    plt.savefig(image_path_recon)


    ##################################################################
    # Saving Results

    if sig_flip:
        if not os.path.exists(os.path.join(parent_dir, "results", "figures","T_estimation","real",  args.noise_type + "_flip", args.model_type)):
            os.makedirs(os.path.join(parent_dir, "results", "figures","T_estimation","real",  args.noise_type + "_flip",  args.model_type))

        if not os.path.exists(os.path.join(parent_dir, "results", "dataframes","T_estimation","real",  args.noise_type + "_flip",  args.model_type)):
            os.makedirs(os.path.join(parent_dir, "results", "dataframes","T_estimation","real",  args.noise_type + "_flip", args.model_type))

        #Saving figure
        image_path_acc = os.path.join(parent_dir, "results","figures", "T_estimation","real", args.noise_type + "_flip", args.model_type, "acc_"+fancy_string+".png")
        image_path_T = os.path.join(parent_dir, "results","figures", "T_estimation","real", args.noise_type + "_flip", args.model_type, "T_"+fancy_string+".png")

        df_path = os.path.join(parent_dir, "results","dataframes", "T_estimation","real", args.noise_type + "_flip",  args.model_type, "df_"+fancy_string+".csv")

    else:
        if not os.path.exists(os.path.join(parent_dir, "results", "figures","T_estimation","real",  args.noise_type, args.model_type)):
            os.makedirs(os.path.join(parent_dir, "results", "figures","T_estimation","real",  args.noise_type,  args.model_type))

        if not os.path.exists(os.path.join(parent_dir, "results", "dataframes","T_estimation","real",  args.noise_type, args.model_type)):
            os.makedirs(os.path.join(parent_dir, "results", "dataframes","T_estimation","real",  args.noise_type,   args.model_type))
    
        #Saving figure
        image_path_acc = os.path.join(parent_dir, "results","figures", "T_estimation","real", args.noise_type, args.model_type, "acc_"+fancy_string+".png")
        image_path_T = os.path.join(parent_dir, "results","figures", "T_estimation","real", args.noise_type,  args.model_type, "T_"+fancy_string+".png")

        df_path = os.path.join(parent_dir, "results","dataframes", "T_estimation","real", args.noise_type, args.model_type, "df_"+fancy_string+".csv")
    #Save DF of results and figure according to noise_type
    
    results_df = {"Accuracy":accuracies, "Noise Frequency":noise_frequencies, "Method":methods, "Estimation Error": T_errors}
    results_df = pd.DataFrame.from_dict(results_df)
    results_df.to_csv(df_path)  

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,5)
    fig = sns.lineplot(data=results_df, x="Noise Frequency", y="Accuracy",hue="Method", err_style='band').set(xlim = (0.00, 0.40))
    plt.savefig(image_path_acc)

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,5)
    fig = sns.lineplot(data=results_df, x="Noise Frequency", y="Estimation Error",hue="Method", err_style='band').set(xlim = (0.00, 0.40))
    plt.savefig(image_path_T)
    