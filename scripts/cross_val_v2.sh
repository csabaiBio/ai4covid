cuda=$1

CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=0 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=1 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=2 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=3 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=4
