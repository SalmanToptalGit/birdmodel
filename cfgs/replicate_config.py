from utils.init_utils import get_device

config = {
    "mel_spec_paramms": {
        "sample_rate": 32000,
        "n_mels": 128,
        "f_min": 40,
        "f_max": 15000,
        "n_fft": 2048,
        "hop_length": 512,
        "normalized": True,
    },
    "backbone": "tf_efficientnetv2_s_in21k",
    "pretrained": True,
    "folds": [0, 1, 2, 3, 4],
    "in_chans": 1,
    "exportable": True,
    "top_db": 80,
    "mixup_p" : 1.0,
    "factor" : 1,

    "train_loader_config": {
        "batch_size": 64,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader_config": {
        "batch_size": 64,
        "num_workers": 8,
        "pin_memory": True,
        "drop_last": False,
    },
    "data_config": {
        "period": 5,
        "kfold": 5,
        "seed": 42,
        "secondary_coef" : 1.0,
        "smooth_label" : 0.1,
    },
    "exp_name": "EXP1",
    "device": get_device(),

    "lr_max": 2.5e-4,
    "lr_min": 1e-7,
    "weight_decay": 1e-6,

    "apex": True,
    "max_grad_norm": 10,


    "base_data_path": "../data",

    "early_stopping": 8,

    "epochs" : 80,
    "output_folder" : "outputs",
}
