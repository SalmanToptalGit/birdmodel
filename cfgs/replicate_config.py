from utils.init_utils import get_device

config = {
    "freq_mask" : {
        "mask_max_length": 10,
        "mask_max_masks": 3,
        "p": 0.3,
        "inplace": True,
    },
    "time_mask" : {
        "mask_max_length": 20,
        "mask_max_masks": 3,
        "p": 0.3,
        "inplace": True,
    },
    "backbone": "eca_nfnet_l0",
    "pretrained": True,
    "folds": [3],
    "in_chans": 3,
    "mixup_p" : 1.0,

    "train_loader_config": {
        "batch_size": 32,
        "num_workers": 8,
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
        "smooth_label" : 0.05,
        "mel_spec_paramms": {
                "sample_rate": 32000,
                "n_mels": 128,
                "f_min": 20,
                "f_max": 16000,
                "n_fft": 2048,
                "hop_length": 512,
                "normalized": True,
                "top_db": 80,
            }
    },
    "exp_name": "EXP3",
    "device": get_device(),

    "lr_max": 3e-4,
    "lr_min": 5e-7,
    "weight_decay": 1e-6,

    "apex": True,
    "max_grad_norm": 10,


    "base_data_path": "../data",

    "early_stopping": 8,

    "epochs" : 200,
    "output_folder" : "outputs",
}
