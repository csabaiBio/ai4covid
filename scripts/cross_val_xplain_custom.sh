cuda=$1

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py --config-name=train_xplain_v2.yaml ++fold=0 ++transformer_heads=16 ++transformer_encode_dim=512 ++cnn_encode_dim=512 ++batch_size=16

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py --config-name=train_xplain_v2.yaml ++fold=1 ++transformer_heads=16 ++transformer_encode_dim=512 ++cnn_encode_dim=512 ++batch_size=16

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py --config-name=train_xplain_v2.yaml ++fold=2 ++transformer_heads=16 ++transformer_encode_dim=512 ++cnn_encode_dim=512 ++batch_size=16

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py --config-name=train_xplain_v2.yaml ++fold=3 ++transformer_heads=16 ++transformer_encode_dim=512 ++cnn_encode_dim=512 ++batch_size=16

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py --config-name=train_xplain_v2.yaml ++fold=4 ++transformer_heads=16 ++transformer_encode_dim=512 ++cnn_encode_dim=512 ++batch_size=16
