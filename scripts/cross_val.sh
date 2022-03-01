cuda=$1


CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=0 ++dataset_identifier=population_average ++img_size=512 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=1 ++dataset_identifier=population_average ++img_size=512 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=2 ++dataset_identifier=population_average ++img_size=512 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=3 ++dataset_identifier=population_average ++img_size=512 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py ++fold=4 ++dataset_identifier=population_average ++img_size=512
