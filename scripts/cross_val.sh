cuda=$1


CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=0 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=1 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=2 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=3 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=4
