cuda=$1

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py ++fold=0 ++transformer_heads=16 ++transformer_encode_dim=256 ++cnn_encode_dim=1024 ++img_size=600

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py ++fold=1 ++transformer_heads=16 ++transformer_encode_dim=256 ++cnn_encode_dim=1024 ++img_size=600

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py ++fold=2 ++transformer_heads=16 ++transformer_encode_dim=256 ++cnn_encode_dim=1024 ++img_size=600

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py ++fold=3 ++transformer_heads=16 ++transformer_encode_dim=256 ++cnn_encode_dim=1024 ++img_size=600

CUDA_VISIBLE_DEVICES=$cuda python run_xplain.py ++fold=4 ++transformer_heads=16 ++transformer_encode_dim=256 ++cnn_encode_dim=1024 ++img_size=600
