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
from data.generate_data import *


def experiment_body(m, n_dims, train_loader, learning_rate, n_states, epochs, length, T_t):
    if m == "Forward_Time_Dependent":
        model = train_RNN_forward(n_dims, train_loader, learning_rate, T_t, output_dim = n_states, EPOCHS = epochs,  verbose=True)
    elif m == "Forward_Time_Independent":
        T_t_basic = np.tile(np.mean(T_t, axis = 0), (length,1,1))
        model = train_RNN_forward(n_dims, train_loader, learning_rate, T_t_basic, output_dim = n_states, EPOCHS = epochs,  verbose=True)
    elif m == "Backward_Time_Dependent":
        model = train_RNN_backward(n_dims, train_loader, learning_rate, T_t, output_dim = n_states, EPOCHS = epochs,  verbose=True)
    elif m == "Backward_Time_Independent":
        T_t_basic = np.tile(np.mean(T_t, axis = 0), (length,1,1))
        model = train_RNN_backward(n_dims, train_loader, learning_rate, T_t_basic, output_dim = n_states, EPOCHS = epochs,  verbose=True)
    elif m == "CE":
        model = train_RNN_CE(n_dims, train_loader, learning_rate, output_dim = n_states, EPOCHS = epochs,  verbose=True)
        
    _, acc, acc_time = evaluate_RNN_time(model, x_test, y_test)

    return model, acc, acc_time


parser = argparse.ArgumentParser('Motivation')

parser.add_argument('--dataset_type', type=str, default='high_var', help="Feature sets to include:")
parser.add_argument('--time_dependency', type=str, default='none', help="Type of time dependent emissions")
parser.add_argument('--n_samples', type =int, default=1000, help="number of samples in synthetic data to generate")
parser.add_argument('--n_dims', type =int, default=50, help="number of dimensions in synthetic data to generate, emissions")
parser.add_argument('--n_states', type =int, default=2, help="number of states/classes in data")
parser.add_argument('--n_iterations', type = int, default = 10, help="number of iterations to train model on dataset")
parser.add_argument('--length', type =int, default=100, help="length (time) of each sample")
parser.add_argument('--noise_type', type=str, default='basic', help="basic, class, or time")
parser.add_argument('--variant', type=str, default='class_independent', help="class independent or dependent noise")
parser.add_argument('--batch_size', type=int, default=256, help="specify batch size")
parser.add_argument('--epochs', type=int, default=150, help="specify num epochs")
parser.add_argument('--learning_rate', type=float, default=0.01, help="specify learning rate")
parser.add_argument('--mask_start_prob', type=float, default=0.9, help="prob of first label being clean")
parser.add_argument('--lam', type=float, default=0.01, help="lambda tradeoff in volmin loss")
parser.add_argument('--a', type=float, default=0.49, help="a parameter for noise function")
parser.add_argument('--b', type=float, default=0.1, help="b parameter for noise function")
parser.add_argument('--center', type=int, default=20, help="timestep to center sigmoid function around")
parser.add_argument('--sig_flip', type=bool, default=False, help="to flip or not to flip sigmoid noise around center y")
parser.add_argument('--add_noise_var', type=float, default=0.1, help="amount of additive noise to add")
parser.add_argument('--model_type', type=str, default="GRU", help="Type of Model")

args = parser.parse_args()

#####################################################################################################

if __name__ == '__main__':

    #experimentID = args.load
    #if experimentID is None:
        # Make a new experiment ID
    #    experimentID = int(SystemRandom().random()*100000)

    start = time.time()

        #check for checkpoint / preemption
    #if not os.path.exists(args.save):
    #    os.makedirs(args.save)

    print('Starting Motivation')
    print("Synthetic Data Type: ", args.dataset_type)
    print("N Samples: ", args.n_samples)
    print("N States: ", args.n_states)
    print("N Dims: ", args.n_dims)
    print("Length: ", args.length)
    print("Noise Type: ", args.noise_type)
    print("Add Noise Var: ", args.add_noise_var)

    dataset_type = args.dataset_type
    n_samples = args.n_samples
    n_states = args.n_states
    n_dims = args.n_dims
    n_iterations = args.n_iterations
    length = args.length
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
    time_dependency = args.time_dependency

    ##################################################################
    filename = generate_filename(n_dims, n_states, n_samples, dataset_type, time_dependency)
    path = "/data/clean/"+time_dependency+"/"
    fancy_string = f"{args.dataset_type}_{args.variant}_n_states_{args.n_states}_n_dims_{args.n_dims}"

    #results path
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    X_clean, Y_clean = load_data(filename,path)

    accuracies = []
    accuracies_time = []
    noise_frequencies = []
    methods = []

    ##################################################################
    #Run Experiments

    if args.noise_type == "basic":
        for a in ([0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]):
            X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                Y_clean, 
                                                args.noise_type, 
                                                n_states, 
                                                a = a,
                                                variant = variant)

            x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size)

            freq = get_flip_frequency(T_t)
            
            for m in ["Forward_Time_Dependent", "Forward_Time_Independent", "CE", "Backward_Time_Dependent", "Backward_Time_Independent"]:
                for iteration in (range(n_iterations)):
                    model, acc, acc_time = experiment_body(m, n_dims, train_loader, learning_rate, n_states, epochs, length, T_t)

                    accuracies.append(acc)
                    accuracies_time.append(acc_time)
                    noise_frequencies.append('{0:.2f}'.format(freq))
                    methods.append(m)

    elif args.noise_type == "exp":
        for b in (reversed([0.004, 0.008, 0.01, 0.015, 0.022, 0.025, 0.03, 0.06, 1.0])):
            if b == 1.0:
                a = 0.0
            else:
                a = 0.49
            X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                Y_clean, 
                                                args.noise_type, 
                                                n_states, 
                                                a = a,
                                                b = b,
                                                variant = variant)

            freq = get_flip_frequency(T_t)
            x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size)

            
            for m in ["Forward_Time_Dependent", "Forward_Time_Independent", "CE", "Backward_Time_Dependent", "Backward_Time_Independent"]:
                for iteration in (range(n_iterations)):
                    model, acc, acc_time= experiment_body(m, n_dims, train_loader, learning_rate, n_states, epochs, length, T_t)

                    accuracies.append(acc)
                    accuracies_time.append(acc_time)
                    noise_frequencies.append('{0:.2f}'.format(freq))
                    methods.append(m)
        

    elif args.noise_type == "sig":
        a_list = [0,   0.2,  0.3, 0.4,  0.45, 0.49, 0.49, 0.49]
        b_list = [0.1,  0.1,  0.1, 0.1,  0.1, 0.1, 0.1, 0.1,]
        c_list = [0,  0, 0, 0,  0.05, 0.1, 0.2, 0.3]

        for a, b, c in zip(a_list, b_list, c_list):
            if b == 1.0:
                a = 0.0
            X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                Y_clean, 
                                                args.noise_type, 
                                                n_states, 
                                                a = a,
                                                b = b,
                                                c = c,
                                                variant = variant)

            freq = get_flip_frequency(T_t)
            x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size)

            
            for m in ["Forward_Time_Dependent", "Forward_Time_Independent", "CE", "Backward_Time_Dependent", "Backward_Time_Independent"]:
                for iteration in (range(n_iterations)):
                    model, acc, acc_time = experiment_body(m, n_dims, train_loader, learning_rate, n_states, epochs, length, T_t)

                    accuracies.append(acc)
                    accuracies_time.append(acc_time)
                    noise_frequencies.append('{0:.2f}'.format(freq))
                    methods.append(m)
                    


    elif args.noise_type == "sin":
        a_list = [0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095]
        b_list = [0, 0.15, 0.2, 0.25, 0.3, 0.33, 0.41, 0.45, 0.49, 0.49, 0.49]
        c_list = [0,  0, 0, 0, 0, 0, 0, 0, 0, 0.1,  0.26]     

        for a, b, c in zip(a_list, b_list, c_list):
            X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                Y_clean, 
                                                args.noise_type, 
                                                n_states, 
                                                a = a,
                                                b = b,
                                                c = c,
                                                variant = variant)

            freq = get_flip_frequency(T_t)
            x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size)

            
            for m in ["Forward_Time_Dependent", "Forward_Time_Independent", "CE", "Backward_Time_Dependent", "Backward_Time_Independent"]:
                for iteration in (range(n_iterations)):
                    model, acc, acc_time = experiment_body(m, n_dims, train_loader, learning_rate, n_states, epochs, length, T_t)
                    accuracies.append(acc)
                    accuracies_time.append(acc_time)
                    noise_frequencies.append('{0:.2f}'.format(freq))
                    methods.append(m)

    elif args.noise_type == "lin":
        a_list = [0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 0.4, 0.35, 0.3, 0.25, 0.2, 0]
        b_list = [0.31, 0.3, 0.2, 0.1, 0.05, 0, 0, 0, 0, 0, 0, 0]

        for a,b in zip(reversed(a_list), reversed(b_list)):
            if a>=b:
                X, Y, Y_tilde, mask, T_t = add_noise(X_clean, 
                                                Y_clean, 
                                                args.noise_type, 
                                                n_states, 
                                                a = a,
                                                b = b,
                                                variant = variant)

                freq = get_flip_frequency(T_t)
                x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

                train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
                train_loader = DataLoader(train_data, batch_size=batch_size)

                
                for m in ["Forward_Time_Dependent", "Forward_Time_Independent", "CE", "Backward_Time_Dependent", "Backward_Time_Independent"]:
                    for iteration in (range(n_iterations)):
                        model, acc, acc_time = experiment_body(m, n_dims, train_loader, learning_rate, n_states, epochs, length, T_t)

                        accuracies.append(acc)
                        accuracies_time.append(acc_time)
                        noise_frequencies.append('{0:.2f}'.format(freq))
                        methods.append(m)

    elif args.noise_type == "mix":
        a = 0.49
        b_list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 1.0]
        mix_a_list = [0.49, 0.49, 0.49, 0.49, 0.45, 0.4, 0.35, 0.3, 0.25, 0]
        mix_c_list = [0.35, 0.3, 0.25, 0.2, 0.1, 0, 0, 0, 0, 0]
        mix_b = 0.1

        for b, mix_a, mix_c in zip(reversed(b_list), reversed(mix_a_list), reversed(mix_c_list)):
            if b == 1.0:
                a = 0.0
            else:
                a = 0.49

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
            x_train, y_train, y_tilde_train, mask_train, x_test, y_test, y_tilde_test, mask_test = train_test_split(X, Y, Y_tilde, mask)

            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_tilde_train),torch.from_numpy(y_train) ,torch.from_numpy(mask_train))
            train_loader = DataLoader(train_data, batch_size=batch_size)

            
            for m in ["Forward_Time_Dependent", "Forward_Time_Independent", "CE", "Backward_Time_Dependent", "Backward_Time_Independent"]:
                for iteration in (range(n_iterations)):
                    model, acc, acc_time = experiment_body(m, n_dims, train_loader, learning_rate, n_states, epochs, length, T_t)

                    accuracies.append(acc)
                    accuracies_time.append(acc_time)
                    noise_frequencies.append('{0:.2f}'.format(freq))
                    methods.append(m)


    ##################################################################
    # Saving Results

    if sig_flip:
        if not os.path.exists(os.path.join(parent_dir, "results", "figures","motivation", args.noise_type + "_flip", args.time_dependency , args.model_type)):
            os.makedirs(os.path.join(parent_dir, "results", "figures","motivation", args.noise_type + "_flip", args.time_dependency,  args.model_type))
        
        if not os.path.exists(os.path.join(parent_dir, "results", "dataframes","motivation", args.noise_type + "_flip", args.time_dependency , args.model_type)):
            os.makedirs(os.path.join(parent_dir, "results", "dataframes","motivation", args.noise_type + "_flip", args.time_dependency,  args.model_type))

        #Saving DF
        image_path = os.path.join(parent_dir, "results","figures", "motivation",args.noise_type + "_flip", args.time_dependency, args.model_type, fancy_string+".png")

        df_path = os.path.join(parent_dir, "results","dataframes", "motivation","synthetic", args.noise_type + "_flip", args.time_dependency, args.model_type, "df_"+fancy_string+".csv")
    else:
        if not os.path.exists(os.path.join(parent_dir, "results", "figures","motivation", args.noise_type, args.time_dependency , args.model_type)):
            os.makedirs(os.path.join(parent_dir, "results", "figures","motivation", args.noise_type, args.time_dependency,  args.model_type))

        if not os.path.exists(os.path.join(parent_dir, "results", "dataframes","motivation","synthetic",  args.noise_type, args.time_dependency , args.model_type)):
            os.makedirs(os.path.join(parent_dir, "results", "dataframes","motivation","synthetic",  args.noise_type, args.time_dependency,  args.model_type))
    
        #Saving DF
        image_path = os.path.join(parent_dir, "results","figures", "motivation",args.noise_type, args.time_dependency, args.model_type, fancy_string+".png")

        df_path = os.path.join(parent_dir, "results","dataframes", "motivation","synthetic", args.noise_type, args.time_dependency, args.model_type, "df_"+fancy_string+".csv")

    #Save DF of results and figure according to noise_type
    
    results_df = {"accuracy":accuracies,"accuracy_time":accuracies_time, "noise_frequency":noise_frequencies, "methods":methods}
    results_df = pd.DataFrame.from_dict(results_df)
    results_df.to_csv(df_path) 

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,5)
    fig = sns.lineplot(data=results_df, x="noise_frequency", y="accuracy",hue="methods", err_style='band', sort= True)
    plt.savefig(image_path)

