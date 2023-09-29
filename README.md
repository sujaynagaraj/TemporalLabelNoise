# TemporalLabelNoise

to run experiments on synthetic data:

python3 -u run_T_estimation.py --dataset_type high_var --variant $1 --noise_type $2 --n_dims $3 --n_states $4 

to run experiments on real data:

--dataset_type: {"HAR", HAR70, EEG_EYE, EEG_SLEEP}
--variant: {"class_conditional", "class_independent"}
python3 -u run_T_estimation_real.py --dataset_type $1 --variant $2 --noise_type $3 --n_states $4 --length $5

