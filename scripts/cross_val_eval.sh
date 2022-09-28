cross_val_dir=$1
gpu=$2

for f in $cross_val_dir/*; do
    if [ -d "$f" ]; then
        # run for test evaluation and validation evaluation
        CUDA_VISIBLE_DEVICES=$gpu python run_inference.py --chkpt_dir $f --save_model --test && CUDA_VISIBLE_DEVICES=$gpu python run_inference.py --chkpt_dir $f
    fi
done