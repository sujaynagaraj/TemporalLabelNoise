# TemporalLabelNoise

to run experiments on synthetic data:

--variant: {"class_independent" "class_conditional"}
--noise_type: {"basic" "mix" "exp" "lin" "sin" "sig"}

python3 -u run_T_estimation.py --dataset_type high_var --variant $1 --noise_type $2 --n_dims 50 --n_states 2 


to run experiments on real data:

--dataset_type: {"HAR" "HAR70" "EEG_EYE" "EEG_SLEEP"}
--variant: {"class_conditional" "class_independent"}
--noise_type: {"basic" "mix" "exp" "lin" "sin" "sig"}

python3 -u run_T_estimation_real.py --dataset_type $1 --variant $2 --noise_type $3 --n_states $4 --length $5


to run experiments on fwd vs bwd:

--variant: {"class_independent" "class_conditional"}
--noise_type: {"basic" "mix" "exp" "lin" "sin" "sig"}

python3 -u run_motivation.py --dataset_type high_var --variant $1 --noise_type $2 --n_dims 50  --n_states 2




all paper figures can be generated with src/paper_figures.ipynb

