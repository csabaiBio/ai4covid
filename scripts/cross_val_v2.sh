cuda=$1


CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=0 ++img_size=512 ++steps_per_epoch=750 ++batch_size=32 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=1 ++img_size=512 ++steps_per_epoch=750 ++batch_size=32 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=2 ++img_size=512 ++steps_per_epoch=750 ++batch_size=32 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=3 ++img_size=512 ++steps_per_epoch=750 ++batch_size=32 &
CUDA_VISIBLE_DEVICES=$cuda python run_hyperopt.py --config-name=train_v2.yaml ++fold=4 ++img_size=512 ++steps_per_epoch=750 ++batch_size=32
