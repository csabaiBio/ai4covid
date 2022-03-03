CUDA_VISIBLE_DEVICES=0 python run_hyperopt.py ++cross_val_train=False ++backbone=EfficientNetB0 &
CUDA_VISIBLE_DEVICES=0 python run_hyperopt.py ++cross_val_train=False ++backbone=EfficientNetB0 &
CUDA_VISIBLE_DEVICES=0 python run_hyperopt.py ++cross_val_train=False ++backbone=EfficientNetB1 &
CUDA_VISIBLE_DEVICES=0 python run_hyperopt.py ++cross_val_train=False ++backbone=EfficientNetB1 &
CUDA_VISIBLE_DEVICES=2 python run_hyperopt.py ++cross_val_train=False ++backbone=EfficientNetB1 &
CUDA_VISIBLE_DEVICES=2 python run_hyperopt.py ++cross_val_train=False ++backbone=ResNet50 &
CUDA_VISIBLE_DEVICES=2 python run_hyperopt.py ++cross_val_train=False ++backbone=ResNet50 &
CUDA_VISIBLE_DEVICES=2 python run_hyperopt.py ++cross_val_train=False ++backbone=ResNet50 &
